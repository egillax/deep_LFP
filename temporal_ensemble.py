from dataset import LFPData
from torch.utils.data import DataLoader
import torch
from models import conv1d_nn
import numpy as np
import pathlib
import os
import visdom
from utils import AverageMeter, calculate_accuracy


def load_model(model_file, sample_length, n_classes):

    # intialize model
    num_channels = 2
    sample_rate = 422
    input_shape = (num_channels, np.int(sample_rate*sample_length))
    model = conv1d_nn.Net(input_shape=input_shape, dropout=0)
    model.fc2 = torch.nn.Linear(2040, n_classes)

    # load previous weights
    state_dict = torch.load(model_file)['state_dict']
    model.load_state_dict(state_dict)

    return model


def load_test_subject(matrix):
    batch_size = 1
    test_set = LFPData(data_file=matrix, split='test', standardize=True)
    test_loader = DataLoader(test_set, shuffle=True, batch_size=batch_size, pin_memory=True,
                             num_workers=1)

    return test_loader, test_set


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # which gpu to us
    torch.backends.cudnn.benchmark = True

    sample_length = 0.5
    num_samples = np.int(np.round(5000/sample_length))
    n_classes = 9

    root_path = pathlib.Path.home() / 'deep_LFP'

    model_file = root_path.joinpath('checkpoints', 'raw_0.5sec_10000_dropout-0_model_best.pth.tar')

    model = load_model(model_file, sample_length=sample_length, n_classes=n_classes)

    data_matrix = root_path.joinpath('data', f'raw2_{sample_length}sec_{num_samples}.npy')
    test_loader, test_set = load_test_subject(data_matrix)
    model.cuda()
    model.eval()

    # true labels are rows, voted labels are columns
    voting_matrix = torch.zeros([n_classes, n_classes], dtype=torch.int32)

    vis = visdom.Visdom(port=8097)
    enviroment = f'bar_subject_prediction_{sample_length}'

    accuracies = AverageMeter()
    for i, data in enumerate(test_loader):
        inputs, labels = data
        labels = labels.cuda(non_blocking=True)
        inputs = inputs.cuda(non_blocking=True)

        outputs = model(inputs)
        accuracy = calculate_accuracy(outputs, labels)
        accuracies.update(accuracy, inputs.size(0))
        vote = outputs.argmax()

        voting_matrix[labels, vote] += 1

        win = [str(x) for x in list(range(n_classes))]
        vis.text(i, win='text_win', env=enviroment)

        for i2, w in enumerate(win):
            vis.bar(voting_matrix[i2, :], win=w, env=enviroment, opts=dict(title=w, rownames=win))
    print(accuracies.avg)


if __name__ == '__main__':
    main()