import os
from os.path import join as opj
import pandas as pd
import numpy as np
import h5py


def load_session_data(session_dir, session_name):
    data_summary_file = [name for name in os.listdir(session_dir) if name.startswith('DataSummary')]
    if not data_summary_file:
        print(session_dir)
        all_data = []
        return all_data
    if len(data_summary_file) > 1:
        my_filter = ['exact timing' in files for files in data_summary_file]
        data_summary_file = [fname for indx, fname in enumerate(data_summary_file) if my_filter[indx] == True][0]
    else:
        data_summary_file = data_summary_file[0]
    data_summary = pd.read_excel(opj(session_dir, data_summary_file), sheet_name='Recording List', header=5)
    data_summary.dropna(axis=1, how='all', inplace=True)
    data_summary.dropna(subset=['File name'], inplace=True)
    data_summary.Notes = data_summary.Notes.astype('str')
    data_summary.sort_values(by=['File name'], inplace=True)
    data_summary = data_summary.reset_index(drop=True)

    if session_name == 'visit_1':
        # create new column with "task"
        start_index = []
        end_index = 0
        task = []
        for ix, notes in enumerate(data_summary.Notes):
            start_bool = any(substring in notes.lower() for substring in ['vas', 'nan', 'end', 'stop'])
            start_bool = not start_bool
            end_bool = any(substring in notes.lower() for substring in ['end', 'stop'])
            if start_bool:
                start_index = ix
                task_name = data_summary.iloc[ix].Notes.lower().replace('start ', '').strip()
                if ' ' in task_name[:]:
                    task_name = task_name[:task_name.index(' ')]
            if end_bool:
                end_index = ix
                start_index = []
            if start_index and end_index:
                task.append(task_name)
            elif end_index == ix:
                task.append(task_name)
            elif start_index == end_index:
                task.append(task_name)
            else:
                task.append('no_task')

        data_summary['Task'] = task

        # locate files I'm interested in, recording from 0-3 contact points
        fnames = data_summary[((data_summary.Ch1 == '0-3') | (data_summary.Ch1 == '3-0')) &
                              (data_summary.Task != 'no_task') &
                              (data_summary.Notes != 'Test') &
                              (~data_summary.Notes.str.lower().str.startswith('Gain'.lower())) &
                              (data_summary.Freq == 422)]['File name'].values
    else:
        strings = 'baseline|Baseline|Wash-out|Resting state'
        fnames = data_summary[((data_summary.Ch1 == '0-3') | (data_summary.Ch1 == '3-0')) &
                              (data_summary.Notes != 'Test') & (data_summary.Freq == 422) &
                              (~data_summary.Notes.str.lower().str.startswith('Gain'.lower())) &
                              (data_summary.Notes.str.contains(strings))]['File name'].values

    all_data = np.empty((0, 2))

    for fname in fnames:
        if fname == 'ocdbd7_2017_03_07_08_07_24__MR_0.xml':
            continue
        txt_file = opj(session_dir, (fname.split('.')[0] + '.txt'))
        data = np.loadtxt(txt_file, dtype=float, delimiter=',')
        data = np.delete(data, [1, 3, 4, 5], axis=1)
        all_data = np.append(all_data, data, axis=0)

    attribute_dict = {'summary_file': data_summary_file}

    return all_data, attribute_dict, data_summary


def clean_hdf5():
    """Clean hdf5 file, remove dbs sessions and empty sessions"""
    dbs_list = '/data/eaxfjord/deep_LFP/dbs_visits.csv'
    df = pd.read_csv(dbs_list)
    dset_names = [(subject + '/' + session) for (subject, session) in zip(df.subjects.values, df.dbs_sessions.values)]
    hdf5_file = '/data/eaxfjord/deep_LFP/all_data_sessions_2.hdf5'

    with h5py.File(hdf5_file, "a") as f:
        for i in range(len(dset_names)):
            print(dset_names[i])
            del f[dset_names[i]]

    with h5py.File(hdf5_file, 'a') as f:
        del f['ocdbd3/Session_2015_11_17_Tuesday']
        del f['ocdbd7/Session_2017_03_06_Monday']
        del f['ocdbd8/Session_2017_11_08_Wednesday']
        del f['ocdbd9/Session_2018_03_07_Wednesday']
        del f['tt/Session_2015_12_01_Tuesday']


def main():
    data_dir = '/data/shared/OCD_DBS_bidirectional/Ruwe_Data/LFP/'

    subjects = [name for name in os.listdir(data_dir) if os.path.isdir(opj(data_dir, name))]

    h5py_file = '/data/eaxfjord/deep_LFP/all_data_sessions_2.hdf5'

    visit_file = '/data/eaxfjord/deep_LFP/visits.csv'
    visits = pd.read_csv(visit_file)

    with h5py.File(h5py_file, "w") as f:

        for subject in subjects:
            subject_dir = opj(data_dir, subject)
            f.create_group(subject)
            sessions = [name for name in os.listdir(subject_dir) if os.path.isdir(opj(subject_dir, name))]
            sessions.sort()
            for session in sessions:
                session_dir = opj(subject_dir, session)
                if visits[visits.subjects == subject].Visit_1.values[0] == session:
                    session_name = 'visit_1'
                else:
                    session_name = session
                session_data, session_attributes = load_session_data(session_dir, session_name)
                print(session_dir, session_data.shape)
                dset = f.create_dataset((subject + '/' + session_name), data=session_data)
                dset.attrs['summary_file'] = session_attributes['summary_file']




if __name__ is "__main__":
    main()
