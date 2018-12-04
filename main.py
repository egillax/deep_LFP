from dataset import LFPData
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


def init_weights(model):
    if type(model) == nn.Conv1d:
        torch.nn.init.kaiming_uniform_(model.weight)


def main():
    best_prec1 = 0
    test = True
    log = True
    save_best = True
    sample_length = 0.5
    num_samples = np.int(np.round(5000/sample_length))  # together I want about 5000 seconds from each subject
    batch_size = 100
    num_epochs = 200
    dropout = 0.4
    task = 'subject_prediction'

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.backends.cudnn.benchmark = True

    root_path = pathlib.Path.cwd()
    matrix = root_path.joinpath('data', f'cleaned_{sample_length}sec_{num_samples}.npy')

    training_dataset = LFPData(data_file=matrix, split='train', standardize=True)
    training_loader = DataLoader(training_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
                                 num_workers=1)

    validation_set = LFPData(data_file=matrix, split='valid', standardize=True)
    validation_loader = DataLoader(validation_set, shuffle=False, batch_size=batch_size, pin_memory=True,
                                   num_workers=1)
    # input_shape = (2, np.int(422 * sample_length))  # this is a hack to figure out shape of fc layer
    # net = conv1d_nn.Net(input_shape=input_shape, dropout=dropout)
    net = conv1d_nn.FCN(in_channels=2, num_classes=9)
    net.apply(init_weights)
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,
                                                     threshold=1e-2)
    stop_criterion = EarlyStopping()

    title = f'FCN2_cleaned_{sample_length}sec_{num_samples}'
    if log:
        log_dir = root_path.joinpath('logs', title)
        if not log_dir.exists():
            log_dir.mkdir()
        training_log = log_dir.joinpath('log')
        if not training_log.exists():
            open(str(training_log), 'w').close()
        result_writer = ResultsWriter(str(training_log), overwrite=True)

    mlog = MeterLogger(server='localhost', port=8097, nclass=9, title=title,
                       env=title)

    for epoch in range(1, num_epochs + 1):
        mlog.timer.reset()

        train_epoch(training_loader, net, criterion, optimizer, mlog)

        if log:
            result_writer.update(title, {'Train': mlog.peek_meter()})
        mlog.print_meter(mode="Train", iepoch=epoch)
        mlog.reset_meter(mode="Train", iepoch=epoch)
        validation_loss = val_epoch(validation_loader, net, criterion, mlog)

        prec1 = mlog.meter['accuracy'].value()[0]

        if save_best:
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            if is_best:
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint(root_path.joinpath('checkpoints', title), {
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': optimizer.state_dict(),
                }, is_best)

        if log:
            result_writer.update(title, {'Validation': mlog.peek_meter()})
        mlog.print_meter(mode="Test", iepoch=epoch)
        mlog.reset_meter(mode="Test", iepoch=epoch)

        stop_criterion.eval_loss(validation_loss)
        if stop_criterion.get_nsteps() >= 30:
            print('Early stopping')
            break
        print(optimizer.param_groups[0]['lr'])
        scheduler.step(validation_loss)

    print('Training finished', best_prec1)

    if test:
        test_set = LFPData(data_file=matrix, split='test', standardize=True)
        test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, pin_memory=True,
                                 num_workers=1)
        test_loss, test_acc = test_epoch(test_loader, net, criterion, mlog)

        result_writer.update(title, {'Test': {'loss': test_loss, 'accuracy': test_acc}})

        print(test_loss, test_acc)

    # save pngs of visdom plot into log path
    plot_visdom(mlog, log_dir)


if __name__ == '__main__':
    main()
