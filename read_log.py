import pandas as pd

pickle_file = '/data/eaxfjord/deep_LFP/logs/LR_4_Layer_1sec_5000_ind_sess_test_set/log'

df = pd.read_pickle(pickle_file)
data = df['results'][0]
train_results = data[-3]
val_results = data[-2]
test_results = data[-1]