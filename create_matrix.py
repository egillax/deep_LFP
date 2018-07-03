""""create matrix with subject labels as first column and 1000 5 sec samples as rows."""
import h5py
import numpy as np
from pytest import File


def combine_subject_data(all_data_file):
    all_data = {}
    with h5py.File(all_data_file, 'r+') as f:
        subjects = list(f.keys())
        subjects.sort()
        for ix, subject in enumerate(subjects):
            sessions = list(f[subject].keys())
            all_subject_data = np.empty((0, 2))
            for session in sessions:
                if session=='Session_2017_03_07_Tuesday': # this session has one channel all zeroes
                    continue
                dset = f[subject + '/' + session].value
                all_subject_data = np.append(all_subject_data, dset, axis=0)
            all_data[subject] = all_subject_data

    return all_data


def create_training_matrix(subject_data, interval=5, sampling_freq=422, num_samples=1000):
    training_matrix = np.empty((len(subject_data) * num_samples, interval * sampling_freq + 1, 2))
    subject_indx = np.zeros((1, 2))
    row_indx = 0
    for key, value in subject_data.items():
        random_index = np.random.randint(0, value.shape[0] - (interval * sampling_freq), num_samples)

        for ix, rand in enumerate(random_index):
            training_matrix[row_indx + ix, :, :] = np.concatenate((subject_indx,
                                                                value[rand:rand + (interval * sampling_freq), :]))
        row_indx += num_samples
        subject_indx += 1

    return training_matrix


def main():
    all_data_file = '/data/eaxfjord/deep_LFP/all_data_sessions_2.hdf5'

    subject_data = combine_subject_data(all_data_file)

    training_matrix = create_training_matrix(subject_data, interval=1)
    training_matrix = np.swapaxes(training_matrix, -1, -2)
    np.random.shuffle(training_matrix)
    np.save('shuffled_LR_1sec.npy', training_matrix)


if __name__ is "__main__":
    main()
