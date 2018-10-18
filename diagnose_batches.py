from dataset import LFPDataStates
from torch.utils.data import DataLoader
import torch
from models import conv1d_nn
import os
from matplotlib import pyplot as plt
from matplotlib import gridspec
import time

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
    nrows=10
    ncols=10
    gs = gridspec.GridSpec(nrows, ncols, wspace=0.0, hspace=0.0,
                           top= 1.-0.5/(nrows+1), bottom=0.5/(nrows+1),
                           left=0.5/(ncols+1), right=1.-0.5/(ncols+1))

    for batch_i, batch in enumerate(training_loader):
        plt.figure()
        ix = 0
        for i in range(nrows):
            for j in range(ncols):
                ax = plt.subplot(gs[i, j])
                ax.plot(batch[0][ix, 0, :].numpy())
                ix += 1





if __name__ == '__main__':
    main()
