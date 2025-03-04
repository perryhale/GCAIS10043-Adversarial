import pandas as pd
import numpy as np


# type: (int, float, float, float, bool) -> tuple[tuple[np.ndarray]]
def get_car_hacking_dataset(
		key,
		train_ratio = 0.7,
		val_ratio = 0.05,
		test_ratio = 0.25, # redundant arg see below
		binary = False
	):
	
	# load data and shuffle
	data = pd.read_csv('car_hacking_dataset/car_hacking_dataset.csv', header=None)
	data = data.sample(frac=1, random_state=key)
	###! truncation for debug and testing
	data = data[:1_000_000]
	
	# optional binary class reduction
	if binary:
		data.iloc[:, -1] = data.iloc[:, -1].apply(lambda y: 1 if y > 1 else y)
	
	# train/val/test split
	train_data = data.iloc[:int(train_ratio*len(data.index)), :]
	val_data = data.iloc[int(train_ratio*len(data.index)):int((train_ratio+val_ratio)*len(data.index)), :]
	test_data = data.iloc[int((train_ratio+val_ratio)*len(data.index)):, :]
	
	# supervised split
	train_x = train_data.iloc[:, :-1].to_numpy()
	train_y = train_data.iloc[:, -1].to_numpy()
	val_x = val_data.iloc[:, :-1].to_numpy()
	val_y = val_data.iloc[:, -1].to_numpy()
	test_x = test_data.iloc[:, :-1].to_numpy()
	test_y = test_data.iloc[:, -1].to_numpy()
	
	return (train_x, train_y), (val_x, val_y), (test_x, test_y)


# type: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
###! assumes: xs in {0..1}
###! when xs.shape == (n,1) and resolution.shape == (1,n) : z.shape == (10, 10) by broadcasting rules
###! causes shape mismatch to propagate
def enforce_res(xs, resolution, mask=None):
	residual = xs - np.round(np.minimum(resolution, np.maximum(0., xs * resolution))) / resolution
	if mask is not None:
		residual *= mask
	return xs - residual


# type: (np.ndarray) -> np.ndarray
def mask_fn(x):
	mask = np.zeros(x.shape)
	mask[2:2+x[1]] = 1.
	return mask
