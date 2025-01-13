import torch.nn.functional as F
import torch
from torch import nn
from my_config import config

class SELayer(nn.Module):
    def __init__(self, leads=59, core_size=2):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(leads, core_size, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(core_size, leads, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, leads, times = x.size()
        y = self.avg_pool(x.transpose(1, 2))
        y = y.view(b, leads)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(b, leads, 1, 1)
        y = y.transpose(1, 2)
        return x * y


class SE_channels_Block(nn.Module):
    def __init__(self, feature_channel, core_num=1):
        super(SE_channels_Block, self).__init__()
        self.feature_channel = feature_channel
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(feature_channel, core_num)
        self.fc2 = nn.Linear(core_num, feature_channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, feature_channel, time_point, channels = x.size()
        squeezed = self.global_avg_pool(x.view(batch_size * feature_channel, 1, time_point, channels))
        squeezed = squeezed.view(batch_size, feature_channel)
        excitation = self.fc1(squeezed)
        excitation = self.fc2(excitation)
        excitation = self.sigmoid(excitation).view(batch_size, feature_channel, 1, 1)
        scale = x * excitation
        return scale + x


class Phased_Encoder(nn.Module):
    def __init__(self, num_channels, o1=config.init_conv_layers, d=config.conv_depth):
        super(Phased_Encoder, self).__init__()
        self.conv1_1 = nn.Conv2d(1, o1 // 3, (1, 32), padding=(0, 16), bias=False)
        self.conv1_2 = nn.Conv2d(1, o1 // 3, (1, 64), padding=(0, 32), bias=False)
        self.conv1_3 = nn.Conv2d(1, o1 // 3, (1, 96), padding=(0, 48), bias=False)
        self.se1 = SELayer(core_size=config.SE_spatial_size)
        self.batchnorm1 = nn.GroupNorm(num_groups=config.GN_groups, num_channels=o1)
        self.depthwiseConv = nn.Conv2d(o1, o1 * d, (num_channels, 1), groups=o1, bias=False)
        self.sec = SE_channels_Block(o1 * d, core_num=config.SE_channels_size)
        self.batchnorm2 = nn.GroupNorm(num_groups=config.GN_groups, num_channels=o1 * d)
        self.elu = nn.GELU()
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.depthwiseConv2_1 = nn.Conv2d(o1 * d, o1 * d, (1, 32), padding=(0, 16), stride=(1, 2), groups=o1 * d, bias=False)
        self.depthwiseConv2_2 = nn.Conv2d(o1 * d, o1 * d, (1, 32), padding=(0, 16), stride=(1, 2), groups=o1 * d, bias=False)
        self.pointwiseConv2 = nn.Conv2d(o1 * d, o1 * d, (1, 1), bias=False)
        self.batchnorm3 = nn.GroupNorm(num_groups=config.GN_groups, num_channels=o1 * d)
        self.elu2 = nn.GELU()
        self.dropout2 = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x3 = self.conv1_3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.se1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseConv(x)
        x = self.sec(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        x = self.dropout1(x)
        x = self.depthwiseConv2_1(x)
        x = self.depthwiseConv2_2(x)
        x = self.pointwiseConv2(x)
        x = self.batchnorm3(x)
        x = self.elu2(x)
        x = self.dropout2(x)
        return x


class TCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, dilation):
        super(TCNLayer, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, dilation=dilation, padding=0)
        self.residual = nn.Conv1d(input_dim, output_dim, kernel_size=1)

    def forward(self, x):
        padding = (self.kernel_size - 1) * self.dilation
        x_padded = F.pad(x, (padding, 0), mode='constant', value=0)
        out = self.conv(x_padded)
        out = F.relu(out)
        residual = self.residual(x)
        out = out + residual
        return out


class CrossAttention(torch.nn.Module):
    def __init__(self, input_dim):
        super(CrossAttention, self).__init__()
        self.W_Q = torch.nn.Linear(input_dim, input_dim)
        self.W_K = torch.nn.Linear(input_dim, input_dim)
        self.W_V = torch.nn.Linear(input_dim, input_dim)

    def forward(self, A, B):
        Q_A = self.W_Q(A)
        K_B = self.W_K(B)
        V_B = self.W_V(B)
        attention_scores = torch.matmul(Q_A, K_B.transpose(-2, -1))
        d_k = Q_A.size(-1)
        scaled_attention_scores = attention_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float))
        attention_weights = F.softmax(scaled_attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V_B)
        return attention_output


class Dynamic_Fusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=3, num_layers=3):
        super(Dynamic_Fusion, self).__init__()
        self.tcn_layers = nn.ModuleList()
        dilation = 1
        for i in range(num_layers):
            self.tcn_layers.append(TCNLayer(input_dim, hidden_dim, kernel_size, dilation))
            input_dim = hidden_dim
            dilation *= config.dilation_expand
        self.fc = nn.Linear(hidden_dim, output_dim)
        input_dim = config.init_conv_layers * config.conv_depth
        self.cross_attention_A = CrossAttention(input_dim=input_dim)
        self.cross_attention_B = CrossAttention(input_dim=input_dim)
        self.inference_mode = False
        self.temporal_weights_1 = []
        self.temporal_weights_2 = []

    def forward(self, feature1, feature2):
        batch_size, feature_layers, _, time_points = feature1.size()
        feature1 = feature1.squeeze(2).permute(0, 2, 1)
        feature2 = feature2.squeeze(2).permute(0, 2, 1)
        feature1 = self.cross_attention_A(feature2, feature1) * feature1 + feature1
        feature2 = self.cross_attention_B(feature1, feature2) * feature2 + feature2
        stage_timepoints = config.stage_timepoints
        padding_length = round(time_points * (200 / stage_timepoints - 1))
        padded_feature1 = torch.cat([feature1, torch.zeros(batch_size, padding_length, feature_layers).to(feature1.device)], dim=1)
        padded_feature2 = torch.cat([torch.zeros(batch_size, padding_length, feature_layers).to(feature2.device), feature2], dim=1)
        dynamic_input = padded_feature1 + padded_feature2
        dynamic_input = dynamic_input.permute(0, 2, 1)
        for tcn_layer in self.tcn_layers:
            dynamic_input = tcn_layer(dynamic_input)
        dynamic_input = dynamic_input.mean(dim=2)
        output = self.fc(dynamic_input)
        self.mmd_sigma = config.mmd_sigma
        loss = self.compute_mmd_loss(feature1, feature2)
        return output, loss

    def compute_mmd_loss(self, feature1, feature2):
        diff1 = feature1.unsqueeze(1) - feature1.unsqueeze(0)
        diff2 = feature2.unsqueeze(1) - feature2.unsqueeze(0)
        diff_cross = feature1.unsqueeze(1) - feature2.unsqueeze(0)
        dist1 = torch.sum(diff1 ** 2, dim=-1)
        dist2 = torch.sum(diff2 ** 2, dim=-1)
        dist_cross = torch.sum(diff_cross ** 2, dim=-1)
        K_XX = torch.exp(-dist1 / (2 * self.mmd_sigma ** 2))
        K_YY = torch.exp(-dist2 / (2 * self.mmd_sigma ** 2))
        K_XY = torch.exp(-dist_cross / (2 * self.mmd_sigma ** 2))
        mmd_loss = torch.mean(K_XX + K_YY - 2 * K_XY)
        return mmd_loss


class PSAFNet(nn.Module):

    def __init__(self, stage_timepoints, lead, time):
        """
        Args:
            stage_timepoints: Number of timepoints for each phase.
            lead: Number of EEG leads (spatial dimension).
            time: Total number of timepoints.
        """
        super(PSAFNet, self).__init__()

        self.lead = lead  # Number of EEG leads (channels)
        self.time = time  # Total timepoints
        self.stage_time = stage_timepoints  # Timepoints per stage (division)

        # Two spatial encoders for the split time stages
        self.time_model1 = Phased_Encoder(lead)
        self.time_model2 = Phased_Encoder(lead)

        # Temporal Convolutional Network for dynamic fusion
        self.TCN_fuse = Dynamic_Fusion(config.init_conv_layers * config.conv_depth, config.TCN_hidden_dim, config.num_class)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, 1 (feature channels), channels, timepoints].

        Returns:
            y_fuse: Final fused feature output.
            similarity_loss: MMD loss between the two extracted feature sets.
        """

        # Split input tensor along the time dimension
        x1 = x[:, :, :, :self.stage_time]  # First half of the timepoints
        x2 = x[:, :, :, self.time - self.stage_time:]  # Second half of the timepoints

        # Extract features from each stage using spatial encoders
        y1 = self.time_model1(x1)  # Features from first stage
        y2 = self.time_model2(x2)  # Features from second stage

        # Fuse features dynamically using TCN and calculate MMD loss
        y_fuse, similarity_loss = self.TCN_fuse(y1, y2)

        return y_fuse, similarity_loss
