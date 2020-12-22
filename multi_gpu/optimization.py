# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python2, python3
"""Functions and classes related to optimization (weight updates)."""

# reference
# https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/python/training/adam.py#L32-L242
# https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/training/optimizer.py#L561
# https://github.com/google-research/albert/blob/master/lamb_optimizer.py
# https://github.com/google-research/albert/blob/master/optimization.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import six
import tensorflow.compat.v1 as tf
from six.moves import zip
from tensorflow.contrib import tpu as contrib_tpu
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu,
                     optimizer="adamw", poly_power=1.0, start_warmup_step=0, clip_norm=1.0, weight_decay_rate=0.01,
                     colocate_gradients_with_ops=False, do_learning_rate_decay=True):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    if do_learning_rate_decay:
        # Implements linear decay of the learning rate.
        learning_rate = tf.train.polynomial_decay(
            learning_rate,
            global_step,
            num_train_steps,
            end_learning_rate=0.0,
            power=poly_power,
            cycle=False)

    # Implements linear warmup. I.e., if global_step - start_warmup_step <
    # num_warmup_steps, the learning rate will be
    # `(global_step - start_warmup_step)/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        tf.logging.info("++++++ warmup starts at step " + str(start_warmup_step)
                        + ", for " + str(num_warmup_steps) + " steps ++++++")
        global_steps_int = tf.cast(global_step, tf.int32)
        start_warm_int = tf.constant(start_warmup_step, dtype=tf.int32)
        global_steps_int = global_steps_int - start_warm_int
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is OK that you use this optimizer for finetuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    # It is OK to use AdamW in the finetuning even the model is trained by LAMB.
    # As report in the Bert pulic github, the learning rate for SQuAD 1.1 finetune
    # is 3e-5, 4e-5 or 5e-5. For LAMB, the users can use 3e-4, 4e-4,or 5e-4 for a
    # batch size of 64 in the finetune.
    if optimizer == "adamw":
        tf.logging.info("using adamw")
        optimizer = AdamWeightDecayOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=weight_decay_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    elif optimizer == "lamb":
        tf.logging.info("using lamb")
        optimizer = LAMBOptimizer(
            learning_rate=learning_rate,
            weight_decay_rate=weight_decay_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    elif optimizer == 'adam':
        tf.logging.info("using adam")
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-6)
    else:
        raise ValueError("Not supported optimizer: ", optimizer)

    if use_tpu:
        optimizer = contrib_tpu.CrossShardOptimizer(optimizer)

    tvars = tf.trainable_variables()
    grads = tf.gradients(
        loss, tvars, colocate_gradients_with_ops=colocate_gradients_with_ops)

    # This is how the model was pre-trained.
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm)

    # train_op = optimizer.apply_gradients(
    #     list(zip(grads, tvars)), global_step=global_step)
    train_op = optimizer.apply_gradients(list(zip(grads, tvars)))

    # Normally the global step update is done inside of `apply_gradients`.
    # However, neither `AdamWeightDecayOptimizer` nor `LAMBOptimizer` do this.
    # But if you use a different optimizer, you should probably take this line
    # out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op, learning_rate, global_step


class AdamWeightDecayOptimizer(tf.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    '''
    Optimizer.apply_gradient method는 multi gpu 학습 지원
    https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/training/optimizer.py#L556

    그러나 apply_gradient 함수를 사용하기 위해서는 아래 methods 구현 필요
    methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse()

    https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/training/optimizer.py#L171
    또한 위 함수에서 _resource_apply_dense, _resource_apply_sparse를 call 하므로 method 구현 필요

    https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/python/training/adam.py
    methods 구현은 위 링크 참조
    '''

    def _create_slots(self, var_list):
        """Create all slots needed by the variables.

        Args:
          var_list: A list of `Variable` objects.
        """
        # No slots needed by default
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _prepare(self):
        """Create all needed tensors before applying gradients.

        This is called with the name_scope using the "name" that
        users have chosen for the application of gradients.
        """

    def _apply_dense(self, grad, var):
        """Add ops to apply dense gradients to `var`.

        Args:
          grad: A `Tensor`.
          var: A `Variable` object.

        Returns:
          An `Operation`.
        """

        assignments = []

        param_name = self._get_variable_name(var.name)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # Standard Adam update.
        next_m = (
                tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
        next_v = (
                tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                          tf.square(grad)))

        update = next_m / (tf.sqrt(next_v) + self.epsilon)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want ot decay the weights in a manner that doesn't interact
        # with the m/v parameters. This is equivalent to adding the square
        # of the weights to the loss with plain (non-momentum) SGD.
        if self._do_use_weight_decay(param_name):
            update += self.weight_decay_rate * var

        update_with_lr = self.learning_rate * update

        next_param = var - update_with_lr

        assignments.extend([var.assign(next_param, use_locking=self._use_locking),
                            m.assign(next_m),
                            v.assign(next_v)])
        return tf.group(*assignments)

    def _resource_apply_dense(self, grad, var):
        """Add ops to apply dense gradients to `var`.

        Args:
          grad: A `Tensor`.
          var: A `Variable` object.

        Returns:
          An `Operation`.
        """

        assignments = []

        param_name = self._get_variable_name(var.name)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # Standard Adam update.
        next_m = (
                tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
        next_v = (
                tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                          tf.square(grad)))

        update = next_m / (tf.sqrt(next_v) + self.epsilon)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want ot decay the weights in a manner that doesn't interact
        # with the m/v parameters. This is equivalent to adding the square
        # of the weights to the loss with plain (non-momentum) SGD.
        if self._do_use_weight_decay(param_name):
            update += self.weight_decay_rate * var

        update_with_lr = self.learning_rate * update

        next_param = var - update_with_lr

        assignments.extend([var.assign(next_param, use_locking=self._use_locking),
                            m.assign(next_m),
                            v.assign(next_v)])
        return tf.group(*assignments)

    def _apply_sparse(self, grad, var):
        """Add ops to apply sparse gradients to `var`.

        The IndexedSlices object passed to `grad` in this function is by default
        pre-processed in `_apply_sparse_duplicate_indices` to remove duplicate
        indices (see its docstring for details). Optimizers which can tolerate or
        have correct special cases for duplicate sparse indices may override
        `_apply_sparse_duplicate_indices` instead of this function, avoiding that
        overhead.

        Args:
          grad: `IndexedSlices`, with no repeated indices.
          var: A `Variable` object.

        Returns:
          An `Operation`.
        """
        return self._apply_sparse_shared(
            grad.values,
            var,
            grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x,
                i,
                v,
                use_locking=self._use_locking))

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        assignments = []

        param_name = self._get_variable_name(var.name)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # Standard Adam update.
        next_m = state_ops.assign(m, m * self.beta_1, use_locking=self._use_locking)
        with ops.control_dependencies([next_m]):
            next_m = scatter_add(m, indices, (grad * (1.0 - self.beta_1)))

        next_v = state_ops.assign(v, v * self.beta_2, use_locking=self._use_locking)
        with ops.control_dependencies([next_v]):
            next_v = scatter_add(v, indices, ((grad * grad) * (1.0 - self.beta_2)))

        update = next_m / (math_ops.sqrt(next_v) + self.epsilon)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want ot decay the weights in a manner that doesn't interact
        # with the m/v parameters. This is equivalent to adding the square
        # of the weights to the loss with plain (non-momentum) SGD.
        if self._do_use_weight_decay(param_name):
            update += self.weight_decay_rate * var

        update_with_lr = self.learning_rate * update

        next_param = state_ops.assign_sub(var, update_with_lr, use_locking=self._use_locking)

        assignments.extend([next_param,
                            next_m,
                            next_v])
        return tf.group(*assignments)

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies([resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        """Add ops to apply sparse gradients to the variable `handle`.
        Similar to `_apply_sparse`, the `indices` argument to this method has been
        de-duplicated. Optimizers which deal correctly with non-unique indices may
        instead override `_resource_apply_sparse_duplicate_indices` to avoid this
        overhead.
        Args:
          grad: a `Tensor` representing the gradient for the affected indices.
          handle: a `Tensor` of dtype `resource` which points to the variable
           to be updated.
          indices: a `Tensor` of integral type representing the indices for
           which the gradient is nonzero. Indices are unique.
        Returns:
          An `Operation` which updates the value of the variable.
        """

        return self._apply_sparse_shared(grad, var, indices, self._resource_scatter_add)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", six.ensure_str(param_name))
        if m is not None:
            param_name = m.group(1)
        return param_name


class LAMBOptimizer(tf.train.Optimizer):
    """LAMB (Layer-wise Adaptive Moments optimizer for Batch training)."""

    # A new optimizer that includes correct L2 weight decay, adaptive
    # element-wise updating, and layer-wise justification. The LAMB optimizer
    # was proposed by Yang You, Jing Li, Jonathan Hseu, Xiaodan Song,
    # James Demmel, and Cho-Jui Hsieh in a paper titled as Reducing BERT
    # Pre-Training Time from 3 Days to 76 Minutes (arxiv.org/abs/1904.00962)

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 exclude_from_layer_adaptation=None,
                 name="LAMBOptimizer"):
        """Constructs a LAMBOptimizer."""
        super(LAMBOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if the
        # arg is None.
        # TODO(jingli): validate if exclude_from_layer_adaptation is necessary.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def _create_slots(self, var_list):
        """Create all slots needed by the variables.

        Args:
          var_list: A list of `Variable` objects.
        """
        # No slots needed by default
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _prepare(self):
        """Create all needed tensors before applying gradients.

        This is called with the name_scope using the "name" that
        users have chosen for the application of gradients.
        """

    def _apply_dense(self, grad, var):
        """Add ops to apply dense gradients to `var`.

        Args:
          grad: A `Tensor`.
          var: A `Variable` object.

        Returns:
          An `Operation`.
        """

        assignments = []

        param_name = self._get_variable_name(var.name)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # Standard Adam update.
        next_m = (
                tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
        next_v = (
                tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                          tf.square(grad)))

        update = next_m / (tf.sqrt(next_v) + self.epsilon)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want ot decay the weights in a manner that doesn't interact
        # with the m/v parameters. This is equivalent to adding the square
        # of the weights to the loss with plain (non-momentum) SGD.
        if self._do_use_weight_decay(param_name):
            update += self.weight_decay_rate * var

        ratio = 1.0
        if self._do_layer_adaptation(param_name):
            w_norm = linalg_ops.norm(var, ord=2)
            g_norm = linalg_ops.norm(update, ord=2)
            ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
                math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

        update_with_lr = ratio * self.learning_rate * update

        next_param = var - update_with_lr

        assignments.extend(
            [var.assign(next_param, use_locking=self._use_locking),
             m.assign(next_m),
             v.assign(next_v)])
        return tf.group(*assignments)

    def _resource_apply_dense(self, grad, var):
        """Add ops to apply dense gradients to `var`.

        Args:
          grad: A `Tensor`.
          var: A `Variable` object.

        Returns:
          An `Operation`.
        """

        assignments = []

        param_name = self._get_variable_name(var.name)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # m = m.handle
        # v = v.handle
        # var = var.handle

        # Standard Adam update.
        next_m = (
                tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
        next_v = (
                tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                          tf.square(grad)))

        update = next_m / (tf.sqrt(next_v) + self.epsilon)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want ot decay the weights in a manner that doesn't interact
        # with the m/v parameters. This is equivalent to adding the square
        # of the weights to the loss with plain (non-momentum) SGD.
        if self._do_use_weight_decay(param_name):
            update += self.weight_decay_rate * var

        ratio = 1.0
        if self._do_layer_adaptation(param_name):
            w_norm = linalg_ops.norm(var, ord=2)
            g_norm = linalg_ops.norm(update, ord=2)
            ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
                math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

        update_with_lr = ratio * self.learning_rate * update

        next_param = var - update_with_lr

        assignments.extend(
            [var.assign(next_param, use_locking=self._use_locking),
             m.assign(next_m),
             v.assign(next_v)])
        return tf.group(*assignments)

    def _apply_sparse(self, grad, var):
        """Add ops to apply sparse gradients to `var`.

        The IndexedSlices object passed to `grad` in this function is by default
        pre-processed in `_apply_sparse_duplicate_indices` to remove duplicate
        indices (see its docstring for details). Optimizers which can tolerate or
        have correct special cases for duplicate sparse indices may override
        `_apply_sparse_duplicate_indices` instead of this function, avoiding that
        overhead.

        Args:
          grad: `IndexedSlices`, with no repeated indices.
          var: A `Variable` object.

        Returns:
          An `Operation`.
        """

        return self._apply_sparse_shared(
            grad.values,
            var,
            grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x,
                i,
                v,
                use_locking=self._use_locking))

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        assignments = []

        param_name = self._get_variable_name(var.name)

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # Standard Adam update.
        next_m = state_ops.assign(m, m * self.beta_1, use_locking=self._use_locking)
        with ops.control_dependencies([next_m]):
            next_m = scatter_add(m, indices, (grad * (1.0 - self.beta_1)))

        next_v = state_ops.assign(v, v * self.beta_2, use_locking=self._use_locking)
        with ops.control_dependencies([next_v]):
            next_v = scatter_add(v, indices, ((grad * grad) * (1.0 - self.beta_2)))

        update = next_m / (math_ops.sqrt(next_v) + self.epsilon)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want ot decay the weights in a manner that doesn't interact
        # with the m/v parameters. This is equivalent to adding the square
        # of the weights to the loss with plain (non-momentum) SGD.
        if self._do_use_weight_decay(param_name):
            update += self.weight_decay_rate * var

        ratio = 1.0
        if self._do_layer_adaptation(param_name):
            w_norm = linalg_ops.norm(var, ord=2)
            g_norm = linalg_ops.norm(update, ord=2)
            ratio = array_ops.where(math_ops.greater(w_norm, 0), array_ops.where(
                math_ops.greater(g_norm, 0), (w_norm / g_norm), 1.0), 1.0)

        update_with_lr = ratio * self.learning_rate * update

        next_param = state_ops.assign_sub(var, update_with_lr, use_locking=self._use_locking)

        assignments.extend([next_param,
                            next_m,
                            next_v])
        return tf.group(*assignments)

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies([resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        """Add ops to apply sparse gradients to the variable `handle`.
        Similar to `_apply_sparse`, the `indices` argument to this method has been
        de-duplicated. Optimizers which deal correctly with non-unique indices may
        instead override `_resource_apply_sparse_duplicate_indices` to avoid this
        overhead.
        Args:
          grad: a `Tensor` representing the gradient for the affected indices.
          handle: a `Tensor` of dtype `resource` which points to the variable
           to be updated.
          indices: a `Tensor` of integral type representing the indices for
           which the gradient is nonzero. Indices are unique.
        Returns:
          An `Operation` which updates the value of the variable.
        """

        return self._apply_sparse_shared(grad, var, indices, self._resource_scatter_add)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", six.ensure_str(param_name))
        if m is not None:
            param_name = m.group(1)
        return param_name
