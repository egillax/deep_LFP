import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class Net(nn.Module):
    def __init__(self, input_shape):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=17, padding=8)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=8, kernel_size=9,
                               stride=1, padding=4)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)

        n_size = self._get_conv_output(input_shape)

        self.fc = nn.Linear(n_size, 9)
        self.softmax = nn.Softmax(dim=0)

    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = F.dropout(self.pool1(F.relu(self.conv1(x))), p=0.5)
        x = F.dropout(self.pool2(F.relu(self.conv2(x))), p=0.5)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.softmax(x)

        return x
