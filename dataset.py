from torch.utils.data import Dataset
import os.path as osp
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class LFP_data(Dataset):

    def __init__(self, root_path, data_file, split, transform=None):
        """
        :param opt: Command line option(/defaults)
        :param split: train | val | test
        :param transform: NotImplemented
        """
        all_data = np.load(osp.join(root_path, data_file))
        y = all_data[:, 0, 0]
        all_data = all_data[:, :, 1:]

        X_train, X_test, y_train, y_test = train_test_split(all_data, y, test_size=900, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=900, stratify=y_train,
                                                          random_state=42)

        if split == 'train':
            data = X_train
            labels = y_train

        if split == 'valid':
            data = X_val
            labels = y_val

        if split == 'test':
            data = X_test
            labels = y_test

        # if opt.standardize:
        #
        #     mask = all_data['mask_3d']
        #
        #     n_subj = data.shape[0]
        #     for i_subj in range(n_subj):
        #         data_subj = data[i_subj]
        #         mean_subj = data_subj[mask].mean(axis=0)
        #         std_subj = data_subj[mask].std(axis=0)
        #         if np.any(std_subj == 0) or np.any(np.isnan(mean_subj)) or np.any(np.isnan(std_subj)):
        #             import pdb;pdb.set_trace()
        #         data[i_subj] = mask * (data_subj - mean_subj) / std_subj

        self.data = torch.from_numpy(data).type(torch.FloatTensor)
        self.labels = torch.from_numpy(labels).type(torch.LongTensor)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


def get_data_set(opt, split, transform=None):

    data_set = LFP_data(opt,
                        split=split,
                        transform=transform)
    return data_set
