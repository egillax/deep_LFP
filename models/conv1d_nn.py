import torch.nn as nn
import torch.nn.functional as F
import torch


class Net(nn.Module):
    def __init__(self, input_shape, dropout):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=9, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, stride=1)
        self.conv4 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1)
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout)
        n_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(n_size, 2040)
        self.fc2 = nn.Linear(2040, 9)

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc2(self.dropout(self.relu(self.fc1(x))))
        return x


class FCN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FCN, self).__init__()
        self.conv1 = Conv1dSame(in_channels=in_channels, out_channels=256, kernel_size=8)
        self.conv1_bn = nn.BatchNorm1d(256)
        self.conv2 = Conv1dSame(in_channels=256, out_channels=128, kernel_size=5)
        self.conv2_bn = nn.BatchNorm1d(128)
        self.conv3 = Conv1dSame(in_channels=128, out_channels=256, kernel_size=3)
        self.conv3_bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.relu(self.conv1_bn(self.conv1(x)))
        x = self.relu(self.conv2_bn(self.conv2(x)))
        x = self.relu(self.conv3_bn(self.conv3(x)))
        x = self.global_avg_pool(x).squeeze(2)
        x = self.fc(x)
        return x


class ResNet(nn.Module):
    def __init__(self, num_classes=9):
        super(ResNet, self).__init__()
        self.res_block1 = ResidualBlock(2, 128, 128)
        self.res_block2 = ResidualBlock(128, 256, 256)
        self.res_block3 = ResidualBlock(256, 256, num_classes)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((num_classes, 1))

    def forward(self, x):
        out = self.res_block1(x)
        out = self.res_block2(out)
        out = self.res_block3(out)
        out = self.global_avg_pool(out).squeeze(2)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv1dSame(in_channels, middle_channels, 8, bias=False)
        self.conv1_bn = nn.BatchNorm1d(middle_channels)
        self.relu = nn.ReLU()
        self.conv2 = Conv1dSame(middle_channels, middle_channels, 5, bias=False)
        self.conv2_bn = nn.BatchNorm1d(middle_channels)
        self.conv3 = Conv1dSame(middle_channels, out_channels, 3, bias=False)
        self.conv3_bn = nn.BatchNorm1d(out_channels)
        self.projection = nn.Sequential(Conv1dSame(in_channels, out_channels, 1, bias=False),
                                        nn.BatchNorm1d(out_channels))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.conv2_bn(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.conv3_bn(out)
        out += self.projection(residual)
        out = self.relu(out)

        return out


class Conv1dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad1d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb)),
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)
