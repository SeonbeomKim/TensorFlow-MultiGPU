import os
from time import time

import numpy as np
import tensorflow as tf
from MultiGPU import MultiGPU
from config import FLAGS
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm


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

        self.flag = FLAGS()
        self.flag.show_flags()

        self.model = MultiGPU(len(self.gpu_list), flag=self.flag)

        self.model.build_placeholders()
        placeholders = self.model.get_placeholders()
        self.model.build_multi_gpu(placeholders, batch_size_per_gpu=self.flag.batch_size)

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

    def restore_model(self, restore_epoch, save_path):
        restore_path = os.path.join(save_path, '%d.ckpt' % restore_epoch)
        self.saver.restore(self.sess, restore_path)
        print('restore: %s' % restore_path)

    def set_tensorboard(self, save_path):
        if not os.path.exists(save_path):
            print("create save directory")
            os.makedirs(save_path)

        with tf.name_scope("tensorboard"):
            self.train_loss_tensorboard = tf.placeholder(tf.float32)
            train_loss_summary = tf.summary.scalar("TRAIN_LOSS", self.train_loss_tensorboard)

            self.val_loss_tensorboard = tf.placeholder(tf.float32)
            val_loss_summary = tf.summary.scalar("VAL_LOSS", self.val_loss_tensorboard)
            self.val_acc_tensorboard = tf.placeholder(tf.float32)
            val_acc_summary = tf.summary.scalar("VAL_ACC", self.val_acc_tensorboard)

            self.test_loss_tensorboard = tf.placeholder(tf.float32)
            test_loss_summary = tf.summary.scalar("TEST_LOSS", self.test_loss_tensorboard)
            self.test_acc_tensorboard = tf.placeholder(tf.float32)
            test_acc_summary = tf.summary.scalar("TEST_ACC", self.test_acc_tensorboard)

            # merged = tf.summary.merge_all()
            self.merged_train = tf.summary.merge([train_loss_summary])
            self.merged_eval = tf.summary.merge(
                [val_loss_summary, val_acc_summary, test_loss_summary, test_acc_summary])
            self.writer = tf.summary.FileWriter(save_path, self.sess.graph)

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

        batch_size = self.flag.batch_size * len(self.gpu_list)
        batch_num = int(np.ceil(len(model_input) / batch_size))
        pbar = tqdm(range(batch_num), dynamic_ncols=True, position=0, mininterval=1.5)
        for i in pbar:
            batch_model_input = model_input[batch_size * i: batch_size * (i + 1)]
            batch_label = label[batch_size * i: batch_size * (i + 1)]

            _, batch_loss, learning_rate = self.sess.run(
                [train_op, loss_op, learning_rate_op],
                {self.model_input: batch_model_input, self.label: batch_label, self.keep_prob: self.flag.keep_prob})

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
        batch_size = self.flag.batch_size
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

    def run(self, save_path, restore_epoch=-1):
        s = time()

        train_set, val_set, test_set = self.load_dataset()
        self.set_model()
        self.set_tensorboard(os.path.join(save_path, 'tensorboard'))
        if restore_epoch > -1:
            self.restore_model(restore_epoch, save_path)

        for epoch in range(restore_epoch+1, self.flag.epoch):
            print('epoch %d' % epoch)
            train_loss = self.train(train_set)

            summary = self.sess.run(self.merged_train, {
                self.train_loss_tensorboard: train_loss})
            self.writer.add_summary(summary, epoch)

            # eval
            val_loss, val_accuracy = self.eval(val_set, desc='val')
            test_loss, test_accuracy = self.eval(test_set, desc='test')
            self.save_model(save_path, epoch)

            summary = self.sess.run(self.merged_eval, {
                self.val_loss_tensorboard: val_loss,
                self.val_acc_tensorboard: val_accuracy,
                self.test_loss_tensorboard: test_loss,
                self.test_acc_tensorboard: test_accuracy})
            self.writer.add_summary(summary, epoch)
            print()

        print('train complete')
        print('gpu_num: %d, time/epoch: %f' % (len(self.gpu_list), (time() - s) / self.flag.epoch))


if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('--gpu_list', type=str)
    args.add_argument('--save_path', type=str, default='./save')
    args.add_argument('--restore_epoch', type=int, default=-1)
    arg_parse = args.parse_args()

    gpu_list = [(gpu.strip()) for gpu in arg_parse.gpu_list.strip(',').split(',')]
    restore_epoch = arg_parse.restore_epoch
    save_path = arg_parse.save_path

    trainer = Trainer(gpu_list=gpu_list)
    trainer.run(save_path, restore_epoch)
