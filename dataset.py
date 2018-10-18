from torch.utils.data import Dataset
import os.path as osp
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class LFPData(Dataset):

    def __init__(self, root_path, data_file, split, transform=None, standardize=None):
        """
        :param opt: Command line option(/defaults)
        :param split: train | val | test
        :param transform: NotImplemented
        """
        all_data = np.load(osp.join(root_path, data_file))
        y = all_data[:, 0, 0]
        all_data = all_data[:, :, 1:]

        test_size = round(all_data.shape[0]*0.1)
        X_train, X_test, y_train, y_test = train_test_split(all_data, y, test_size=test_size, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, stratify=y_train,
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

        if standardize:

            n_samples = data.shape[0]
            for i_sample in range(n_samples):
                data_sample = data[i_sample]
                mean_sample = data_sample.mean(axis=1, keepdims=True)
                std_sample = data_sample.std(axis=1, keepdims=True)
                if np.any(std_sample == 0) or np.any(np.isnan(mean_sample)) or np.any(np.isnan(std_sample)):
                    assert(std_sample == 0)
                data[i_sample] = (data_sample - mean_sample) / std_sample

        self.data = torch.from_numpy(data).type(torch.FloatTensor)
        self.labels = torch.from_numpy(labels).type(torch.LongTensor)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


class LFPDataStates(Dataset):

    def __init__(self, root_path, data_file, split, transform=None, standardize=None):

        all_data = np.load(data_file)
        training_data = np.swapaxes(all_data['training'], -2, -1)
        test_data = np.swapaxes(all_data['test'], -2, -1)

        y = training_data[:, 0, 1]
        subject_id = training_data[:, 0, 0]
        subjects = np.unique(subject_id)

        #pick a random subject for use as validation subject
        validation_subject = np.random.randint(0, len(subjects))
        training_subjects = [x for x in subjects if x != validation_subject]


        if split == 'train':
            data = training_data[np.logical_or.reduce([training_data[:, 0, 0] == x for x in training_subjects]), :, 2:]
            labels  = training_data[np.logical_or.reduce([training_data[:, 0, 0] == x for x in training_subjects]), 0, 1]
        if split == 'valid':
            data = training_data[training_data[:, 0, 0] == validation_subject, :, 2:]
            labels = training_data[training_data[:,0,0] == validation_subject, 0, 1]
        if split == 'test':
            data = test_data[:, :, 2:]
            labels = test_data[:, 0, 1]

        if standardize:

            n_samples = data.shape[0]
            for i_sample in range(n_samples):
                data_sample = data[i_sample]
                mean_sample = data_sample.mean(axis=1, keepdims=True)
                std_sample = data_sample.std(axis=1, keepdims=True)
                if np.any(std_sample == 0) or np.any(np.isnan(mean_sample)) or np.any(np.isnan(std_sample)):
                    assert(std_sample == 0)
                data[i_sample] = (data_sample - mean_sample) / std_sample

        self.data = torch.from_numpy(data).type(torch.FloatTensor)
        self.labels = torch.from_numpy(labels).type(torch.LongTensor)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item], self.labels[item]