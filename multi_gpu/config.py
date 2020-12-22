class FLAGS:
    def __init__(self):
        self.batch_size = 512 # each gpu
        self.height = 28
        self.width = 28
        self.channel = 1
        self.label_num = 10

        self.init_lr = 1e-3

        self.epoch = 10 # for time check
        self.num_train_steps = 100000
        self.num_warmup_steps = 100
        self.keep_prob = 0.9
        self.save_step = 500

        self.optimizer = 'lamb'
        self.clip_norm = 1.0
        self.colocate_gradients_with_ops = True

        self.log_step_count_steps = 100 # estimator train info step
    def show_flags(self):
        print('flags:')
        for key, value in self.__dict__.items():
            print('%s: %s' % (key, value))
        print()
