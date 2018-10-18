""""create matrix with subject labels as first column and samples as rows."""
import h5py
import numpy as np


def combine_subject_data(all_data_file):
    all_data = {}
    with h5py.File(all_data_file, 'r+') as f:
        subjects = list(f.keys())
        subjects.sort()
        for ix, subject in enumerate(subjects):
            sessions = list(f[subject].keys())
            all_subject_data = np.empty((0, 3))
            for ix2, session in enumerate(sessions):
                if session == 'Session_2017_03_07_Tuesday':  # this session has one channel all zeroes
                    continue
                dset = f[subject + '/' + session].value
                dset = np.append(dset, ix2 * np.ones((dset.shape[0], 1)), axis=1)
                all_subject_data = np.append(all_subject_data, dset, axis=0)
            all_data[subject] = all_subject_data

    return all_data


def create_training_matrix(subject_data, interval=5, sampling_freq=422, num_samples=1000):
    test_samples = np.int(0.1 * num_samples)
    training_samples = np.int(0.9 * num_samples)
    training_matrix = np.empty((len(subject_data) * training_samples, interval * sampling_freq + 1, 2))
    test_matrix = np.empty((len(subject_data) * test_samples, interval * sampling_freq +1, 2))
    subject_indx = np.zeros((1, 2))
    train_row_indx = 0
    test_row_indx = 0

    for key, value in subject_data.items():
        session_ids = np.unique(value[:, 2])
        session_sizes = [np.where(value[:, 2] == i)[0].shape for i in session_ids]
        test_session_id = session_sizes.index(min(session_sizes))
        train_session_id = np.where(session_ids != test_session_id)

        train_set = value[np.isin(value[:, 2], train_session_id[0]), :2]
        test_set = value[value[:, 2] == test_session_id, :2]

        train_random_index = np.random.randint(0, train_set.shape[0] - (interval * sampling_freq), training_samples)
        test_random_index = np.random.randint(0, test_set.shape[0] - (interval * sampling_freq), test_samples)

        for ix, rand in enumerate(train_random_index):
            training_matrix[train_row_indx + ix, :, :] = np.concatenate((subject_indx,
                                                                   train_set[rand:rand + (interval * sampling_freq), :]))
        for ix, rand in enumerate(test_random_index):
            test_matrix[test_row_indx + ix, :, :] = np.concatenate((subject_indx,
                                                                    test_set[rand:rand + (interval * sampling_freq), :]))

        train_row_indx += training_samples
        test_row_indx += test_samples
        subject_indx += 1

    return training_matrix, test_matrix


def main():
    all_data_file = '/data/eaxfjord/deep_LFP/all_data_sessions_2.hdf5'

    subject_data = combine_subject_data(all_data_file)

    training_matrix, test_matrix = create_training_matrix(subject_data, interval=1, num_samples=5000)
    all_matrix = np.concatenate((training_matrix, test_matrix), axis=0)

    all_matrix = np.swapaxes(all_matrix, -1, -2)
    np.random.shuffle(training_matrix)
    np.save('all_shuffled_LR_1sec_5000.npy', all_matrix)


if __name__ is "__main__":
    main()
