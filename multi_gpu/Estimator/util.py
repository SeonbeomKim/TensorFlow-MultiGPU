import os

def set_visible_gpu_divices(gpu_list):
    set_gpu_allow_growth() # 이게 먼저 실행되어야만 allow_growth 워킹함.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(','.join(gpu_list))


def set_gpu_allow_growth():
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
