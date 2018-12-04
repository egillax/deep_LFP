from dataset import LFPDataStates, LFPDataStatesPercentSplit
from torch.utils.data import DataLoader
import torch
from models import conv1d_nn
import torch.optim as optim
import torch.nn as nn
import os
from utils import EarlyStopping, save_checkpoint, plot_visdom
from train import train_epoch, val_epoch, test_epoch
from my_meterlogger import MeterLogger
from torchnet.utils import ResultsWriter
import numpy as np
import pathlib
import visdom
import json


def main():
    best_prec1 = 0
    test = True
    transfer_learning = True
    batch_size = 50
    sample_length = 3
    num_epochs = 50
    task = 'state_prediciton'

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.backends.cudnn.benchmark = True

    root_path = pathlib.Path.home().joinpath('deep_LFP')
    matrix = root_path.joinpath('data', f'cleaned_state_matrix_{sample_length}sec.npz')

    training_dataset = LFPDataStates(data_file=matrix, split='train', standardize=True)
    training_loader = DataLoader(training_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
                                 num_workers=1)

    validation_set = LFPDataStates(data_file=matrix, split='valid', standardize=True)
    validation_loader = DataLoader(validation_set, shuffle=False, batch_size=batch_size, pin_memory=True,
                                   num_workers=1)
    input_shape = (2, np.int(422 * sample_length))  # this is a hack to figure out shape of fc layer
    net = conv1d_nn.Net(input_shape=input_shape, dropout=0)
    if transfer_learning:
        num_samples_prev_model = np.int(np.round(5000/sample_length))
        previous_model = f'cleaned_{sample_length}sec_{num_samples_prev_model}_model_best.pth.tar'
        previous_model_weights = os.path.join(root_path, 'checkpoints', previous_model)
        net.load_state_dict(torch.load(previous_model_weights)['state_dict'])
        for param in net.parameters():
            param.requires_grad = False

        num_features = net.fc1.in_features
        net.fc1 = nn.Linear(num_features, 2040)
    net.fc2 = nn.Linear(2040, 4)
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100,
                                                     threshold=1e-3)
    stop_criterion = EarlyStopping()

    title = f'cleaned_state_prediction_{sample_length}sec_transfer_learning'
    training_log_path = '/data/eaxfjord/deep_LFP/logs/' + title + '/log'
    log_dir = os.path.dirname(training_log_path)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(training_log_path):
        open(training_log_path, 'w').close()

    result_writer = ResultsWriter(training_log_path, overwrite=True)

    mlog = MeterLogger(server='localhost', port=8097, nclass=4, title=title,
                       env=f'state_prediction_{sample_length}sec')

    for epoch in range(1, num_epochs + 1):
        mlog.timer.reset()

        train_epoch(training_loader, net, criterion, optimizer, mlog)

        result_writer.update(task, {'Train': mlog.peek_meter()})
        mlog.print_meter(mode="Train", iepoch=epoch)
        mlog.reset_meter(mode="Train", iepoch=epoch)
        validation_loss = val_epoch(validation_loader, net, criterion, mlog)

        prec1 = mlog.meter['accuracy'].value()[0]

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint(os.path.join(root_path, 'checkpoints', title), {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

        result_writer.update(task, {'Validation': mlog.peek_meter()})

        mlog.print_meter(mode="Test", iepoch=epoch)
        mlog.reset_meter(mode="Test", iepoch=epoch)

        # stop_criterion.eval_loss(validation_loss)
        # if stop_criterion.get_nsteps() >= 30:
        #     print('Early stopping')
        #     break
        print(optimizer.param_groups[0]['lr'])
        scheduler.step(validation_loss)

    print('Training finished', best_prec1)

    if test:
        test_set = LFPDataStates(data_file=matrix, split='test', standardize=True)
        test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, pin_memory=True,
                                 num_workers=1)
        test_loss, test_acc = test_epoch(test_loader, net, criterion, mlog)

        result_writer.update(task, {'Test': test_acc})
        print(test_loss, test_acc)

    # when finished get data from visdom plot, and save to png
    plot_visdom(mlog, log_dir)


if __name__ == '__main__':
    main()
