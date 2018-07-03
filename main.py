from dataset import LFP_data
from torch.utils.data import DataLoader
import torch
from models import conv1d_nn
import torch.optim as optim
import torch.nn as nn
import os
from utils import EarlyStopping, save_checkpoint
from train import train_epoch
from validation import val_epoch
from my_meterlogger import MeterLogger
from torchnet.utils import ResultsWriter


def main():
    best_prec1 = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.backends.cudnn.benchmark = True

    root_path = '/data/eaxfjord/deep_LFP'
    matrix = 'shuffled_LR_1sec.npy'
    batch_size = 20

    training_dataset = LFP_data(root_path=root_path, data_file=matrix, split='train', standardize=True)
    training_loader = DataLoader(training_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
                                 num_workers=1)

    validation_set = LFP_data(root_path=root_path, data_file=matrix, split='valid', standardize=True)
    validation_loader = DataLoader(validation_set, shuffle=False, batch_size=batch_size, pin_memory=True,
                                   num_workers=1)
    length = 10
    input_shape = (2, 422*length)  # this is a hack to figure out shape of fc layer
    net = conv1d_nn.Net(input_shape=input_shape, dropout=0.4)
    net.cuda()

    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100,
                                                     threshold=1e-3)
    num_epochs = 60
    stop_criterion = EarlyStopping()

    title = 'LR_4_Layer_1sec'
    training_log_path = '/data/eaxfjord/deep_LFP/logs/' + title + '/log'
    base_dir = os.path.dirname(training_log_path)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if not os.path.exists(training_log_path):
        open(training_log_path, 'w').close()

    result_writer = ResultsWriter(training_log_path, overwrite=True)

    mlog = MeterLogger(server='localhost', port=8097, nclass=9, title=title)

    for epoch in range(1, num_epochs+1):
        mlog.timer.reset()

        train_epoch(training_loader, net, criterion, optimizer, mlog)

        result_writer.update('early_stopping', {'Train': mlog.peek_meter()})
        mlog.print_meter(mode="Train", iepoch=epoch)
        mlog.reset_meter(mode="Train", iepoch=epoch)
        validation_loss = val_epoch(validation_loader, net, criterion, mlog)

        prec1 = mlog.meter['accuracy'].value()[0]

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(title, {
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        result_writer.update('early_stopping', {'Test': mlog.peek_meter()})
        mlog.print_meter(mode="Test", iepoch=epoch)
        mlog.reset_meter(mode="Test", iepoch=epoch)

        # stop_criterion.eval_loss(validation_loss)
        # if stop_criterion.get_nsteps() >= 30:
        #     print('Early stopping')
        #     break
        print(optimizer.param_groups[0]['lr'])
        scheduler.step(validation_loss)

    print('Training finished', best_prec1)


if __name__ == '__main__':
    main()