"""Clean hdf5 file, remove dbs sessions and empty sessions"""
import h5py
import pandas as pd

dbs_list = '/data/eaxfjord/deep_LFP/dbs_visits.csv'
df = pd.read_csv(dbs_list)

dset_names = [(subject+ '/' + session) for (subject, session) in zip(df.subjects.values, df.dbs_sessions.values)]


hdf5_file = '/data/eaxfjord/deep_LFP/all_data_sessions.hdf5'

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
