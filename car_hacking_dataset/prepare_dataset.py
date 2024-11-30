import numpy as np
import pandas as pd
from tqdm import tqdm; tqdm.pandas()

"""https://ocslab.hksecurity.net/Datasets/car-hacking-dataset"""

def correct_label(row):
	dlc = row[2]
	flag = row[3+dlc]
	row[3+dlc] = np.nan
	row[11] = flag
	return row

df_norm = pd.read_csv('normal_dataset.csv', header=None, names=range(12)).progress_apply(correct_label, axis=1)
df_dos = pd.read_csv('DoS_dataset.csv', header=None, names=range(12)).progress_apply(correct_label, axis=1)
df_fuzz = pd.read_csv('Fuzzy_dataset.csv', header=None, names=range(12)).progress_apply(correct_label, axis=1)
df_gear = pd.read_csv('gear_dataset.csv', header=None, names=range(12)).progress_apply(correct_label, axis=1)
df_rpm = pd.read_csv('RPM_dataset.csv', header=None, names=range(12)).progress_apply(correct_label, axis=1)

df_norm[11] = df_norm[11].map(lambda x : 0)
df_dos[11] = df_dos[11].map(lambda x : 0 if (x=='R') else 1)
df_fuzz[11] = df_fuzz[11].map(lambda x : 0 if (x=='R') else 2)
df_gear[11] = df_gear[11].map(lambda x : 0 if (x=='R') else 3)
df_rpm[11] = df_rpm[11].map(lambda x : 0 if (x=='R') else 4)

df_conc = pd.concat([df_norm, df_dos, df_fuzz, df_gear, df_rpm])
df_conc = df_conc.drop(columns=0)
df_conc[[1,3,4,5,6,7,8,9,10]] = df_conc[[1,3,4,5,6,7,8,9,10]].fillna('0').map(lambda x : int(x, 16))

with open('car_hacking_dataset.csv', 'w') as f:
	for i,row in df_conc.iterrows():
		f.write(','.join([str(x) for x in row]) + '\n')
