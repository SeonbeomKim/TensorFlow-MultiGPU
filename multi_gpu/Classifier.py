from collections import namedtuple

import tensorflow as tf

from multi_gpu.Model import SimpleCNN as Model
from multi_gpu.optimization import create_optimizer
import multi_gpu.config as config

FLAGS = config.FLAGS()


class Classifier(Model):
    def __init__(self, *args, **kwargs):
        super(Classifier, self).__init__(*args, **kwargs)
        self.scope = 'Classifier'

    def build_placeholders(self):
        super(Classifier, self).build_placeholders()
        self.label = tf.placeholder(tf.float32, [None, 10])

    def build_model(self, model_input, keep_prob):
        super(Classifier, self).build_model(model_input, keep_prob)
        o = self.get_output_layer()
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.predict = tf.layers.dense(o, units=FLAGS.label_num, activation=None)

    def build_loss(self, predict, label):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=label,
            logits=predict))

    def build_optimizer(self, loss, colocate_gradients_with_ops=True):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.train_op, self.learning_rate_op, self.global_step_op = \
                create_optimizer(loss,
                                 FLAGS.init_lr,
                                 FLAGS.num_train_steps,
                                 FLAGS.num_warmup_steps,
                                 use_tpu=False,
                                 optimizer=FLAGS.optimizer,
                                 clip_norm=FLAGS.clip_norm,
                                 colocate_gradients_with_ops=colocate_gradients_with_ops)

    def build_metric(self, predict, label):
        self.metric = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(predict, 1), tf.argmax(label, 1)), tf.int32))
        self.accuracy = tf.metrics.accuracy(labels=tf.argmax(label, 1),
                                                predictions=tf.argmax(predict, 1))

    def get_placeholders(self):
        placeholders = super(Classifier, self).get_placeholders()
        model_input = placeholders.model_input
        keep_prob = placeholders.keep_prob

        data_format = namedtuple('placeholders', ['model_input', 'keep_prob', 'label'])
        return data_format(model_input, keep_prob, self.label)

    def get_predict(self):
        return self.predict

    def get_loss(self):
        return self.loss

    def get_metric(self):
        return self.metric

    def get_accuracy(self):
        return self.accuracy

    def get_train_op(self):
        return self.train_op

    def get_learning_rate_op(self):
        return self.learning_rate_op

    def get_global_step_op(self):
        return self.global_step_op
