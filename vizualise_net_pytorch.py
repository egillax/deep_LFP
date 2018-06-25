import re
import torch
from torch.autograd import Variable
from models import conv1d_nn
from torchviz import make_dot



inputs = Variable(torch.randn(1, 1, 2110))
input_shape = (1, 2110)
net = conv1d_nn.Net(input_shape=input_shape)
y = net(inputs)
# print(y)

g = make_dot(y.mean(), params=dict(net.named_parameters()))
g.view()