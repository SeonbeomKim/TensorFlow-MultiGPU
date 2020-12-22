import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

class generator:
    def __init__(self, mode, batch_size, epoch):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        if mode == 'train':
            self._input = mnist.train.images
            self._label = mnist.train.labels
        elif mode == 'val':
            self._input = mnist.validation.images
            self._label = mnist.validation.labels
        elif mode == 'test':
            self._input = mnist.test.images
            self._label = mnist.test.labels
        else:
            raise Exception('mode: train, val or test')

        self.batch_size = batch_size
        self.epoch = epoch

    def __call__(self):
        for _ in range(self.epoch):
            for i in range(int(np.ceil(len(self._input) / self.batch_size))):
                _input = self._input[self.batch_size * i: self.batch_size * (i + 1)]
                _label = self._label[self.batch_size * i: self.batch_size * (i + 1)]
                yield _input, _label


def input_fn_builder(mode, batch_size, epoch):
    keys = ['input', 'label']

    def _parser(*record):
        features = {}
        for key, _record in zip(keys, record):
            features[key] = _record
        return features

    def input_fn():
        Generator = tf.data.Dataset.from_generator(generator(mode=mode, batch_size=batch_size, epoch=epoch),
                                                   output_types=(tf.float32, tf.int32),
                                                   output_shapes=((None, 784), (None, 10)))

        # Generator = Generator.repeat(epoch)
        # Generator = Generator.shuffle(10000)
        # Generator = Generator.batch(batch_size)
        Generator = Generator.map(_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        Generator = Generator.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return Generator

    return input_fn