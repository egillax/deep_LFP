import csv
import pandas as pd
from sklearn import metrics
import torch
import numpy as np
import nitime.algorithms as tsa
import shutil
import warnings
import visdom
import json
import matplotlib.pyplot as plt
from matplotlib import gridspec
import itertools as it



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


class EarlyStopping():
    def __init__(self):
        self.nsteps_similar_loss = 0
        self.previous_loss = 9999.0
        self.delta_loss = 1e-3

    def _increment_step(self):
        self.nsteps_similar_loss += 1

    def _reset(self):
        self.nsteps_similar_loss = 0

    def eval_loss(self, loss):
        if (self.previous_loss - loss) <= self.delta_loss:
            self._increment_step()
            self.previous_loss = loss
        else:
            self._reset()
            self.previous_loss = loss

    def get_nsteps(self):
        return self.nsteps_similar_loss


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def plot_visdom(meterlogger, log_dir):
    """recreate visdom plots using matplotlib and save as png in log folder"""

    viz = visdom.Visdom()
    window_json = viz.get_window_data(env=meterlogger.env)
    window_data = json.loads(window_json)

    for plot in window_data.keys():
        if plot in ['loss', 'accuracy']:
            plot_content = window_data[plot]['content']['data']
            training_data = plot_content[1]
            test_data = plot_content[2]

            x_train, y_train = training_data['x'], training_data['y']
            x_test, y_test = test_data['x'], test_data['y']

            plt.ioff()
            plt.figure()
            plt.plot(x_train, y_train, label=training_data['name'])
            plt.plot(x_test, y_test, label=test_data['name'])
            plt.legend()
            plt.title(plot)
            plt.savefig(f'{log_dir}/{plot}.png')
        elif 'confusion' in plot:
            plot_content = window_data[plot]['content']['data']
            fig, ax = plt.subplots()
            im = ax.imshow(plot_content[0]['z'], origin='lower')
            ax.set_xticks(plot_content[0]['x'])
            ax.set_yticks(plot_content[0]['y'])
            ax.set_xlabel('predicted classes')
            ax.set_ylabel('true classes')
            ax.set_title(plot)
            fig.colorbar(im)
            fig.savefig(f'{log_dir}/{plot}.png')
        else:
            warnings.warn(f'Unknown plot type: {plot}')


def plot_from_log(log_file):
    """plot data from pickled log file"""
    df = pd.read_csv(log_file, delimiter='\t')

    epochs = df.epoch.values
    loss = df.loss.values
    acc = df.acc.values

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('loss', color=color)
    ax1.plot(epochs, loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('acc', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, acc, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size


def calc_multiclass_metrics(outputs, targets):

    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    _, predicted = torch.max(outputs, 1)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        accuracy_score = metrics.accuracy_score(targets, predicted)
        precision = metrics.precision_score(targets, predicted, average='macro', labels=labels)
        recall = metrics.recall_score(targets, predicted, average='macro', labels=labels)
        f1_score = metrics.f1_score(targets, predicted, average='macro', labels=labels)
        fbeta = metrics.fbeta_score(targets, predicted, average='macro', beta=0.5, labels=labels)

    return accuracy_score, precision, recall, f1_score, fbeta


def plot_frequency_spectrum(matrix):
    sampling_freq = 422

    for i in range(10):
        # rows = np.where(training_matrix[:, 0] == 8.0)
        rand_row = np.random.randint(0, matrix.shape[0])
        print(rand_row)
        timeseries = matrix[rand_row, 1:]
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax1.plot(timeseries)
        f, fx, _ = tsa.spectral.multi_taper_psd(timeseries, Fs=sampling_freq)
        ax2.plot(f, fx)
        plt.show()


def save_checkpoint(title, state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    dst = f'{str(title)}_model_best.pth.tar'
    if is_best:
        shutil.copy(filename, dst)


def plot_kernel(model):
    kernel = model.conv1.weight.data.clone()

    nrows=8
    ncols=8
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.0, hspace=0.0,
                           top= 1.-0.5/(nrows+1), bottom=0.5/(nrows+1),
                           left=0.5/(ncols+1), right=1.-0.5/(ncols+1))
    ix = 0
    for i,j in it.product(range(nrows),range(ncols)):
        ax = plt.subplot(gs[i, j])
        ax.plot(kernel[ix, ...].cpu().numpy().T)
        ix += 1