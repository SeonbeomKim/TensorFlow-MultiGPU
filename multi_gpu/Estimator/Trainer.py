import tensorflow as tf
from multi_gpu.Estimator.util import set_visible_gpu_divices

tf.logging.set_verbosity(tf.logging.INFO)

from multi_gpu.Classifier import Classifier
from multi_gpu.Estimator.BatchGenerator import input_fn_builder
import multi_gpu.config as config
from time import time

FLAGS = config.FLAGS()


def model_fn_builder():
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        model_input = features["input"]
        label = features["label"]

        model = Classifier()

        if mode == tf.estimator.ModeKeys.TRAIN:
            model.build_model(model_input=model_input,
                              keep_prob=FLAGS.keep_prob)
            predict = model.get_predict()
            model.build_loss(predict=predict,
                             label=label)
            loss = model.get_loss()

            model.build_optimizer(loss=loss,
                                  colocate_gradients_with_ops=True)
            train_op = model.get_train_op()

            # train info 수정 가능, tf.estimator.RunConfig에서 log_step_count_steps=None 설정 필요
            logging_hook = tf.compat.v1.train.LoggingTensorHook({'loss': loss,
                                                                 'lr': model.get_learning_rate_op(),
                                                                 'step': model.get_global_step_op()},
                                                                every_n_iter=FLAGS.log_step_count_steps)
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op,
                                              training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:
            model.build_model(model_input=model_input,
                              keep_prob=1.0)
            predict = model.get_predict()
            model.build_loss(predict=predict,
                             label=label)
            loss = model.get_loss()

            model.build_metric(predict=predict,
                               label=label)
            accuracy = model.get_accuracy()
            eval_metric_ops = {'ACC': accuracy}

            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              eval_metric_ops=eval_metric_ops)

    return model_fn


def main(gpu_list, save_path):
    if not gpu_list:
        raise Exception('ERROR: no gpu list')

    gpu_list = sorted(list(set(gpu_list)), reverse=False)
    set_visible_gpu_divices(gpu_list)

    '''
    train_distribute 사용할 때는 not working
    'TF_FORCE_GPU_ALLOW_GROWTH' true 설정 필요
    set_visible_gpu_devices 안에 작성해뒀음. 
     
    # session_config = tf.ConfigProto()
    # session_config.gpu_options.allow_growth = True 
    '''
    session_config = None

    train_gpu_devices = ["/gpu:%d" % i for i in range(len(gpu_list))]
    train_distribution_strategy = tf.distribute.MirroredStrategy(devices=train_gpu_devices)

    run_config = tf.estimator.RunConfig(model_dir=save_path,
                                        save_checkpoints_steps=10000,  # FLAGS.save_step,  # 모델 저장 및 evaluate 실행
                                        save_summary_steps=10000,  # 300,  # train loss tensorboard 저장
                                        keep_checkpoint_max=10000,
                                        session_config=session_config,
                                        log_step_count_steps=None,
                                        train_distribute=train_distribution_strategy,
                                        eval_distribute=None)

    estimator = tf.estimator.Estimator(model_fn=model_fn_builder(),
                                       config=run_config)

    train_input_fn = input_fn_builder('train', batch_size=FLAGS.batch_size, epoch=10)
    val_input_fn = input_fn_builder('val', batch_size=FLAGS.batch_size, epoch=1)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=FLAGS.num_train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=val_input_fn,
                                      throttle_secs=10)  # evaluation 간격 시간, 이것보다 짧은 간격에 eval call되면 eval 안함.

    tf.estimator.train_and_evaluate(estimator, train_spec=train_spec, eval_spec=eval_spec)


if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('--gpu_list', type=str)
    args.add_argument('--save_path', type=str, default='./save')
    arg_parse = args.parse_args()

    gpu_list = [(gpu.strip()) for gpu in arg_parse.gpu_list.strip(',').split(',')]
    save_path = arg_parse.save_path

    s = time()

    main(gpu_list=gpu_list, save_path=save_path)

    print('train complete')
    print('gpu_num: %d, sec/epoch: %f' % (len(gpu_list), (time() - s) / FLAGS.epoch))
