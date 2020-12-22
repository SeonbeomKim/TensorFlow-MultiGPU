# Tensorflow MultiGPU Example(Estimator)

## Tensorflow
    tensorflow-gpu 1.4.0

## Run(example)
    python Trainer.py --gpu_list=2,3,4 --save_path={str}

## Experiments
    TITAN X:
        gpu_num: 1, sec/epoch 9.646515
        gpu_num: 2, sec/epoch: 6.923067

## Reference
    https://github.com/tensorflow/tensorflow/blob/v2.4.0/tensorflow/python/training/adam.py#L32-L242
    https://github.com/tensorflow/tensorflow/blob/v1.14.0/tensorflow/python/training/optimizer.py#L561
    https://github.com/google-research/albert/blob/master/lamb_optimizer.py
    https://github.com/google-research/albert/blob/master/optimization.py
    https://github.com/google-research/albert/blob/master/run_pretraining.py
    https://www.tensorflow.org/guide/distributed_training?hl=ko#%EC%B6%94%EC%A0%95%EA%B8%B0estimator%EC%99%80_%ED%95%A8%EA%BB%98_tfdistributestrategy_%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0    
