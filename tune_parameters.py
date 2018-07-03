from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from models import conv1d_nn
from dataset import LFP_data
from torch.utils.data import DataLoader
import torch
import os
import torch.optim as optim
import torch.nn as nn
from train import train_epoch
from validation import val_epoch

count = 0


def objective(dropout):
    global count
    count += 1
    print('-------------------------------------------------------------------')
    print('%d' % count)
    print(dropout)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.backends.cudnn.benchmark = True

    root_path = '/data/eaxfjord/deep_LFP'
    matrix = 'shuffled_LR.npy'
    batch_size = 20

    training_dataset = LFP_data(root_path=root_path, data_file=matrix, split='train')
    training_loader = DataLoader(training_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
                                 num_workers=1)

    validation_set = LFP_data(root_path=root_path, data_file=matrix, split='valid')
    validation_loader = DataLoader(validation_set, shuffle=False, batch_size=batch_size, pin_memory=True,
                                   num_workers=1)

    input_shape = (2, 2110)  # this is a hack to figure out shape of fc layer
    net = conv1d_nn.Net(input_shape=input_shape, dropout=dropout)
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100,
                                                     threshold=1e-3)
    num_epochs = 200

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch(training_loader, net, criterion, optimizer)
        validation_loss, validation_accuracy = val_epoch(validation_loader, net, criterion)

        scheduler.step(validation_loss)
        print('EPOCH:: %i, (%s), train_loss, test_loss: %.3f, train_acc: %.3f, test_acc: %.3f' % (epoch + 1,
                                                                                               train_loss,
                                                                                               validation_loss,
                                                                                               train_acc,
                                                                                               validation_accuracy))

    return {'loss': -validation_accuracy, 'status': STATUS_OK, 'val_loss': validation_loss}


trials = Trials()
parameter_space = hp.uniform('dropout', 0.1, 1)

best = fmin(objective,
            space=parameter_space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print(best)
