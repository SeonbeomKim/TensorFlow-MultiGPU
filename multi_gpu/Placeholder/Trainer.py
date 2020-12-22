import os

import multi_gpu.config as config
import numpy as np
import tensorflow as tf
from multi_gpu.Placeholder.MultiGPU import MultiGPU
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
from time import time

FLAGS = config.FLAGS()


class Trainer:
    def __init__(self, gpu_list=[]):
        self.gpu_list = sorted(list(set(gpu_list)), reverse=False)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(','.join(self.gpu_list))

    def load_dataset(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        train_set = (mnist.train.images, mnist.train.labels)
        val_set = (mnist.validation.images, mnist.validation.labels)
        test_set = (mnist.test.images, mnist.test.labels)
        return train_set, val_set, test_set

    def set_model(self):
        FLAGS.show_flags()

        self.model = MultiGPU(len(self.gpu_list))

        self.model.build_placeholders()
        placeholders = self.model.get_placeholders()
        self.model.build_multi_gpu(placeholders, batch_size_per_gpu=FLAGS.batch_size)

        tensor_config = tf.ConfigProto()
        tensor_config.gpu_options.allow_growth = True

        self.saver = tf.train.Saver(max_to_keep=10000)
        self.sess = tf.Session(config=tensor_config)
        self.sess.run(tf.global_variables_initializer())

        self.model_input = placeholders.model_input
        self.keep_prob = placeholders.keep_prob
        self.label = placeholders.label

        print('trainable_variables')
        print(tf.trainable_variables())

    def save_model(self, save_path, epoch):
        if not os.path.exists(save_path):
            print("create save directory")
            os.makedirs(save_path)

        save_path = os.path.join(save_path, '%d.ckpt' % epoch)
        self.saver.save(self.sess, save_path)
        print('save model: %s' % save_path)


    def train(self, dataset):
        total_loss = 0

        model_input, label = dataset

        loss_op = self.model.get_loss()
        train_op = self.model.get_train_op()
        learning_rate_op = self.model.get_learning_rate_op()

        ## remove for pure train time check
        # indices = np.arange(len(model_input))
        # np.random.shuffle(indices)
        # model_input = model_input[indices]
        # label = label[indices]

        batch_size = FLAGS.batch_size * len(self.gpu_list)
        batch_num = int(np.ceil(len(model_input) / batch_size))
        pbar = tqdm(range(batch_num), dynamic_ncols=True, position=0, mininterval=1.5)

        for i in pbar:
            batch_model_input = model_input[batch_size * i: batch_size * (i + 1)]
            batch_label = label[batch_size * i: batch_size * (i + 1)]

            _, batch_loss, learning_rate = self.sess.run(
                [train_op, loss_op, learning_rate_op],
                {self.model_input: batch_model_input, self.label: batch_label, self.keep_prob: FLAGS.keep_prob})

            total_loss += batch_loss
            pbar.update(1)
            pbar.set_description("[train] loss: %f, lr: %0.8f" % (total_loss / (i + 1), learning_rate), refresh=False)
        pbar.close()

        return total_loss / batch_num

    def eval(self, dataset, desc='val'):
        total_loss, total_correct, label_num = 0, 0, 0

        loss_op = self.model.get_loss()
        correct_metric_op = self.model.get_metric()

        model_input, label = dataset
        batch_size = FLAGS.batch_size
        batch_num = int(np.ceil(len(model_input) / batch_size))
        pbar = tqdm(range(batch_num), dynamic_ncols=True, position=1, mininterval=1.5)
        for i in pbar:
            batch_model_input = model_input[batch_size * i: batch_size * (i + 1)]
            batch_label = label[batch_size * i: batch_size * (i + 1)]

            batch_loss, batch_correct = self.sess.run([loss_op, correct_metric_op],
                                                      {self.model_input: batch_model_input, self.label: batch_label,
                                                       self.keep_prob: 1.0})

            total_loss += batch_loss
            total_correct += batch_correct
            label_num += len(batch_label)

            pbar.set_description("[%s] loss: %f, acc: %0.8f" % (desc, total_loss / (i + 1), total_correct / label_num),
                                 refresh=False)

        return total_loss / batch_num, total_correct / label_num

    def run(self, save_path):
        train_set, val_set, test_set = self.load_dataset()
        self.set_model()

        for epoch in range(1, FLAGS.epoch):
            print('epoch %d' % epoch)
            train_loss = self.train(train_set)
            print('train_loss: %f\n' % train_loss)

        # eval
        val_loss, val_accuracy = self.eval(val_set, desc='val')
        print('val_loss: %f, val_accuracy: %f' % (val_loss, val_accuracy))
        self.save_model(save_path, FLAGS.epoch)



if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('--gpu_list', type=str)
    args.add_argument('--save_path', type=str, default='./save')
    arg_parse = args.parse_args()

    gpu_list = [(gpu.strip()) for gpu in arg_parse.gpu_list.strip(',').split(',')]
    save_path = arg_parse.save_path

    trainer = Trainer(gpu_list=gpu_list)

    s = time()

    trainer.run(save_path)

    print('train complete')
    print('gpu_num: %d, sec/epoch: %f' % (len(gpu_list), (time() - s) / FLAGS.epoch))
