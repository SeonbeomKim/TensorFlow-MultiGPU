from collections import namedtuple

import tensorflow as tf


class SimpleCNN:
    def __init__(self, flag, seed=777):
        tf.set_random_seed(seed)
        self.flag = flag

    def build_placeholders(self):
        self.model_input = tf.placeholder(tf.float32, [None, 784])
        self.keep_prob = tf.placeholder_with_default(1.0, name='keep_prob', shape=[])

    def build_model(self, model_input, keep_prob):
        with tf.variable_scope('SimpleCNN', reuse=tf.AUTO_REUSE):
            model_input = tf.reshape(model_input, (-1, self.flag.height, self.flag.width, self.flag.channel))

            layer1 = tf.layers.conv2d(model_input, filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME',
                                      activation=tf.nn.relu)

            pool_layer1 = tf.layers.max_pooling2d(layer1, pool_size=[2, 2], strides=[2, 2], padding='SAME')
            drop_layer1 = tf.nn.dropout(pool_layer1, keep_prob=keep_prob)

            layer2 = tf.layers.conv2d(drop_layer1, filters=256, kernel_size=[3, 3], strides=[1, 1], padding='SAME',
                                      activation=tf.nn.relu)
            pool_layer2 = tf.layers.max_pooling2d(layer2, pool_size=[2, 2], strides=[2, 2], padding='SAME')
            drop_layer2 = tf.nn.dropout(pool_layer2, keep_prob=keep_prob)

            layer3 = tf.layers.conv2d(drop_layer2, filters=512, kernel_size=[3, 3], strides=[1, 1], padding='SAME',
                                      activation=tf.nn.relu)
            pool_layer3 = tf.layers.max_pooling2d(layer3, pool_size=[2, 2], strides=[2, 2], padding='SAME')
            drop_layer3 = tf.nn.dropout(pool_layer3, keep_prob=keep_prob)

            self.output = tf.layers.flatten(drop_layer3)  # feature

    def get_placeholders(self):
        data_format = namedtuple('placeholders', ['model_input', 'keep_prob'])
        return data_format(self.model_input, self.keep_prob)

    def get_output_layer(self):
        return self.output
