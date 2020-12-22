# Tensorflow MultiGPU Example(Placeholder)

## Tensorflow
    tensorflow-gpu 1.4.0

## Run(example)
    python Trainer.py --gpu_list=2,3,4 --save_path={str}

## Experiments
    TITAN X:
        gpu_num: 1, sec/epoch: 8.555405
        gpu_num: 2, sec/epoch: 6.200181

## Reference
    https://github.com/google-research/albert/blob/master/lamb_optimizer.py
    https://github.com/google-research/albert/blob/master/optimization.py
    https://wizardforcel.gitbooks.io/tensorflow-examples-aymericdamien/content/6.2_multigpu_cnn.html
    
