from load_data_hdf5 import load_session_data
import h5py
import pandas as pd
from os.path import join as opj
import numpy as np

#
# def get_states(session, subject):
#     pass()


def main():
    data_dir = '/data/shared/OCD_DBS_bidirectional/Ruwe_Data/LFP/'
    hdf5_file = '/data/eaxfjord/deep_LFP/data/all_sessions_cleaned.hdf5'
    with h5py.File(hdf5_file, 'r') as f:
        subjects = list(f)

    visit_file = '/data/eaxfjord/deep_LFP/data/visits.csv'
    visits = pd.read_csv(visit_file)

    subject_dict = {}
    for subject in subjects:
        visit_1 = visits[visits.subjects == subject].Visit_1.values[0]
        session_dir = opj(data_dir, subject, visit_1)
        _, _, summary = load_session_data(session_dir, 'visit_1')

        data_dict = {}
        tasks = summary.Task.unique()
        task_dict = {}
        for task in tasks:
            if task == 'no_task':
                continue
            elif task == 'compulsies':
                task = 'compulsions'

            fnames = summary[((summary.Ch1 == '0-3') | (summary.Ch1 == '3-0')) & (summary.Task == task)]['File name'].values

            all_data = np.empty((0, 2))
            for fname in fnames:
                txt_file = opj(session_dir, (fname.split('.')[0] + '.txt'))
                data = np.loadtxt(txt_file, dtype=float, delimiter=',')
                data = np.delete(data, [1, 3, 4, 5], axis=1)
                all_data = np.append(all_data, data, axis=0)

            task_dict[task] = all_data
        subject_dict[subject] = task_dict
        np.save('cleaned_task_dictionary.npy', subject_dict)

if __name__ is "__main__":
    main()
