import tensorflow as tf

from Classifier import Classifier as TASK


class MultiGPU(TASK):
    def __init__(self, gpu_num, *args, **kwargs):
        self.gpu_num = gpu_num
        self.args = args
        self.kwargs = kwargs

    def build_multi_gpu(self, placeholders, batch_size_per_gpu):
        # with tf.device('/cpu:0'):
        loss_list = []

        for i in range(self.gpu_num):
            with tf.device('/gpu:%d' % i):
                # Split data between GPUs
                _model_input = placeholders.model_input[i * batch_size_per_gpu: (i + 1) * batch_size_per_gpu]
                _label = placeholders.label[i * batch_size_per_gpu: (i + 1) * batch_size_per_gpu]

                TASK.__init__(self, *self.args, **self.kwargs)
                self.build_model(_model_input, placeholders.keep_prob)
                predict = self.get_predict()

                self.build_loss(predict, _label)
                loss = TASK.get_loss(self)
                loss_list.append(loss)

                # Only first GPU compute metric
                if i == 0:
                    self.build_metric(predict, _label)
                    self.correct_metric = self.get_metric()

        loss_list = tf.boolean_mask(loss_list, ~tf.is_nan(loss_list))  # except Nan value
        self.multi_gpu_loss = tf.reduce_mean(loss_list)
        self.build_optimizer(self.multi_gpu_loss, colocate_gradients_with_ops=True)

    def get_loss(self):
        return self.multi_gpu_loss
