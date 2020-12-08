# Tensorflow MultiGPU Example

## Tensorflow
    tensorflow-gpu 1.4.0

## run(example)
    python Trainer.py --gpu_list=2,3,4 --save_path={str} --restore_epoch={int}

## Experiments
    TITAN X:
        gpu_num: 1, sec/epoch: 13.19
        gpu_num: 2, sec/epoch: 10.14
        gpu_num: 3, sec/epoch: 8.23
        gpu_num: 4, sec/epoch: 6.83

## reference
    https://github.com/google-research/albert/blob/master/lamb_optimizer.py
    https://github.com/google-research/albert/blob/master/optimization.py
    https://wizardforcel.gitbooks.io/tensorflow-examples-aymericdamien/content/6.2_multigpu_cnn.html
    
