from dataset import LFPDataStates
from torch.utils.data import DataLoader
import torch
from models import conv1d_nn
import torch.optim as optim
import torch.nn as nn
import os
from utils import EarlyStopping, save_checkpoint
from train import train_epoch, val_epoch, test_epoch
from my_meterlogger import MeterLogger
from torchnet.utils import ResultsWriter


def main():
    best_prec1 = 0
    test = False
    state_prediction = True

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.backends.cudnn.benchmark = True

    root_path = '/data/eaxfjord/deep_LFP'
    matrix = 'state_matrix.npz'
    batch_size = 100

    training_dataset = LFPDataStates(root_path=root_path, data_file=matrix, split='train', standardize=True)
    training_loader = DataLoader(training_dataset, shuffle=True, batch_size=batch_size, pin_memory=True,
                                 num_workers=1)

    validation_set = LFPDataStates(root_path=root_path, data_file=matrix, split='valid', standardize=True)
    validation_loader = DataLoader(validation_set, shuffle=False, batch_size=batch_size, pin_memory=True,
                                   num_workers=1)
    length = 1
    input_shape = (2, 422 * length)  # this is a hack to figure out shape of fc layer
    net = conv1d_nn.Net(input_shape=input_shape, dropout=0)
    if state_prediction:
        previous_model_weights = '/data/eaxfjord/deep_LFP/LR_4_Layer_1sec_5000_not_ind_sess_test_set_fc_dropout_model_best.pth.tar'
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
    num_epochs = 200
    stop_criterion = EarlyStopping()

    title = 'first_try_with_state_prediction'
    training_log_path = '/data/eaxfjord/deep_LFP/logs/' + title + '/log'
    base_dir = os.path.dirname(training_log_path)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if not os.path.exists(training_log_path):
        open(training_log_path, 'w').close()

    result_writer = ResultsWriter(training_log_path, overwrite=True)

    mlog = MeterLogger(server='localhost', port=8097, nclass=4, title=title)

    for epoch in range(1, num_epochs + 1):
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

    if test:
        test_set = LFPDataStates(root_path=root_path, data_file=matrix, split='test', standardize=True)
        test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, pin_memory=True,
                                 num_workers=1)
        test_loss, test_acc = test_epoch(test_loader, net, criterion, mlog)

        print(test_loss, test_acc)


if __name__ == '__main__':
    main()
