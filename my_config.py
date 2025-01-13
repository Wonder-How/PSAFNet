class Config():
    def __init__(self):

        # training parameters
        self.seed = 42
        self.num_channels = 59
        self.num_timepoints = 200
        self.num_epochs = 5
        self.learning_rate = 0.001
        self.batchsize = 16
        self.fs = 200
        self.num_class = 2

        # PSAFNet parameters
        self.stage_timepoints = 150
        self.init_conv_layers = 12
        self.conv_depth = 2
        self.SE_spatial_size = 2
        self.SE_channels_size = 1
        self.GN_groups = 3
        self.dropout_rate = 0.2

        self.dilation_expand = 2
        self.mmd_sigma = 1.0
        self.TCN_hidden_dim = 24


        # Print all parameters
        self.print_config()

    def print_config(self):
        print("Configuration parameters:")
        for key, value in vars(self).items():
            print(f"{key}: {value}")



config = Config()