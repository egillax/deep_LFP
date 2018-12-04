import numpy as np
import pathlib

def create_training_matrix(data, interval=1, sampling_freq=422, num_samples=100):
    # exclude subject because it's missing a state
    del data['ocdbd8']

    subjects = list(data.keys())
    num_subjects = len(subjects)
    num_states = 4
    states = ['baseline', 'obsessions', 'compulsions', 'relief']

    # select test subject
    test_subject = subjects[np.random.randint(0, 7)]

    training_subjects = [x for x in subjects if x != test_subject]
    training_matrix = np.empty(((len(training_subjects) * num_samples * num_states), np.int(interval * sampling_freq) + 2, 2))
    test_matrix = np.empty((num_samples * num_states, np.int(interval * sampling_freq) + 2, 2))

    # training dataset
    train_row_indx = 0
    for sub_indx, subject in enumerate(training_subjects):
        for state_idx, state in enumerate(states):
            # select random samples
            state_size = data[subject][state].shape[0]
            state_data = data[subject][state]
            train_random_index = np.random.randint(0, state_size - (interval * sampling_freq), num_samples)

            for ix, rand in enumerate(train_random_index):
                training_matrix[train_row_indx + ix, :, :] = np.concatenate((np.array([[sub_indx, sub_indx]]),
                                                                             np.array([[state_idx, state_idx]]),
                                                                             state_data[rand:rand +
                                                                                        np.int(interval * sampling_freq),
                                                                             :]))
            train_row_indx += num_samples

    # test dataset
    test_row_indx = 0
    for state_idx, state in enumerate(states):
        state_size = data[test_subject][state].shape[0]
        state_data = data[test_subject][state]

        test_random_indx = np.random.randint(0, state_size - (interval * sampling_freq), num_samples)

        for ix, rand in enumerate(test_random_indx):
            test_matrix[test_row_indx+ix, :, :] = np.concatenate((np.array([[1, 1]]), np.array([[state_idx, state_idx]]),
                                                    state_data[rand:rand + np.int(interval * sampling_freq), :]))
        test_row_indx += num_samples

    name = f'cleaned_state_matrix_{interval}sec.npz'
    path = pathlib.Path.cwd().joinpath('data', name)
    np.savez(path, training=training_matrix, test=test_matrix)


def main():
    task_dictionary = np.load('cleaned_task_dictionary.npy').item()

    create_training_matrix(task_dictionary, interval=3, sampling_freq=422, num_samples=33)


if __name__ is "__main__":
    main()
