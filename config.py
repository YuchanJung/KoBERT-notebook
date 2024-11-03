class Config:
    def __init__(self):
        self.max_len = 64
        self.batch_size = 64
        self.warmup_ratio = 0.1
        self.num_epochs = 12
        self.max_grad_norm = 1
        self.log_interval = 200
        self.learning_rate =  5e-5

config = Config()