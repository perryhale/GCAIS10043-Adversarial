import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from sklearn.tree import DecisionTreeClassifier

import art
from art.attacks.evasion import DecisionTreeAttack
from art.estimators.classification import SklearnClassifier

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, accuracy_score


### set RNG seed
RNG_SEED = 0
np.random.seed(RNG_SEED)


### data preparation
TRAIN_RATIO = 0.7
VAL_RATIO = 0.05
TEST_RATIO = 1 - TRAIN_RATIO + VAL_RATIO
FEATURE_SCALE = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255])

# type: (np.ndarray) -> np.ndarray
def mask_fn(x):
	mask = np.zeros(x.shape)
	mask[2:2+x[1]] = 1.
	return mask

# type: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
def enforce_res(xs, res, mask=None):
	res = xs - np.round(np.minimum(res, np.maximum(0., xs * res))) / res
	if mask is not None:
		res *= mask
	return xs - res

# type: (pd.DataFrame, float, float) -> tuple[pd.DataFrame]
def train_val_test_split(data):
	train_data = data.iloc[:int(TRAIN_RATIO*len(data.index)), :]
	val_data = data.iloc[int(TRAIN_RATIO*len(data.index)):int((TRAIN_RATIO+VAL_RATIO)*len(data.index)), :]
	test_data = data.iloc[int((TRAIN_RATIO+VAL_RATIO)*len(data.index)):, :]
	return (train_data, val_data, test_data)

# type: (pd.DataFrame) -> tuple[np.ndarray]
def standard_split(data):
	x = data.iloc[:, :-1].to_numpy() / FEATURE_SCALE
	y = data.iloc[:, -1].to_numpy()
	return (x, y)

# type: (pd.DataFrame) -> tuple[tuple[np.ndarray]]
def adversarial_split(data):
	data_ben = data.loc[data[10] == 0]
	data_mal = data.loc[data[10] >= 1]
	ben_x = data_ben.iloc[:, :-1].to_numpy() / FEATURE_SCALE
	ben_y = data_ben.iloc[:, -1].to_numpy()
	mal_x = data_mal.iloc[:, :-1].to_numpy() / FEATURE_SCALE
	mal_y = data_mal.iloc[:, -1].to_numpy()
	mal_yt = np.zeros(len(mal_y))
	ben_mask = np.apply_along_axis(mask_fn, axis=1, arr=data_ben.iloc[:, :-1].to_numpy())
	mal_mask = np.apply_along_axis(mask_fn, axis=1, arr=data_mal.iloc[:, :-1].to_numpy())
	return ((ben_x, ben_y, ben_mask), (mal_x, mal_y, mal_yt, mal_mask))

# load data and shuffle
data = pd.read_csv('car_hacking_dataset/car_hacking_dataset.csv', header=None)
data = data.sample(frac=1)

# basic partition
train_data, val_data, test_data = train_val_test_split(data[:10_000])

# train set oversample and standard split
train_x, train_y = standard_split(train_data)
train_x, train_y = RandomOverSampler().fit_resample(train_x, train_y)

# val standard split 
val_x, val_y = standard_split(val_data)

# test set adversarial split
test_ben, test_mal = adversarial_split(test_data)#[:100_000]) # truncation of test set
(test_ben_x, test_ben_y, test_ben_mask) = test_ben
(test_mal_x, test_mal_y, test_mal_yt, test_mal_mask) = test_mal

print(train_x.shape, train_y.shape, 'train')
print(val_x.shape, val_y.shape, 'validation')
print(test_ben_x.shape, test_ben_y.shape, 'test (benign)')
print(test_mal_x.shape, test_mal_y.shape, 'test (malicious)')


### evaluate tree depths
DEPTH_MIN = 3
DEPTH_MAX = 100
DEPTH_INC = 1
VERBOSE = True

depth_axis = range(DEPTH_MIN, DEPTH_MAX+DEPTH_INC, DEPTH_INC)
grid_history = {
	'test_ben_acc':[], 
	'test_mal_acc':[], 
	'test_ben_adv_acc':[], 
	'test_mal_adv_acc':[], 
	'test_mal_adv_acct':[], 
	'test_ben_cfm':[], 
	'test_mal_cfm':[], 
	'test_ben_adv_cfm':[], 
	'test_mal_adv_cfm':[]}

for depth in tqdm(depth_axis):
	
	# train model
	model = DecisionTreeClassifier(max_depth=depth) # uses numpy RNG backend
	model.fit(train_x, train_y)
	
	# setup art wrapper
	art_model = SklearnClassifier(model)
	dta_untargeted = DecisionTreeAttack(art_model, verbose=VERBOSE)
	dta_targeted = DecisionTreeAttack(art_model, verbose=VERBOSE)
	
	# generate samples
	test_ben_adv_x = enforce_res(dta_untargeted.generate(test_ben_x, mask=test_ben_mask), FEATURE_SCALE)
	test_mal_adv_x = enforce_res(dta_targeted.generate(test_mal_x, test_mal_yt, mask=test_mal_mask), FEATURE_SCALE)
	
	# evaluate model on samples
	test_ben_yh = model.predict(test_ben_x)
	test_mal_yh = model.predict(test_mal_x)
	test_ben_adv_yh = model.predict(test_ben_adv_x)
	test_mal_adv_yh = model.predict(test_mal_adv_x)
	
	test_ben_acc = accuracy_score(test_ben_y, test_ben_yh)
	test_mal_acc = accuracy_score(test_mal_y, test_mal_yh)
	test_ben_adv_acc = accuracy_score(test_ben_adv_yh, test_ben_y)
	test_mal_adv_acc = accuracy_score(test_mal_adv_yh, test_mal_y)
	test_mal_adv_acct = accuracy_score(test_mal_adv_yh, test_mal_yt)
	
	test_ben_cfm = confusion_matrix(test_ben_y, test_ben_yh, labels=range(5))
	test_mal_cfm = confusion_matrix(test_mal_y, test_mal_yh, labels=range(5))
	test_ben_adv_cfm = confusion_matrix(test_ben_y, test_ben_adv_yh, labels=range(5))
	test_mal_adv_cfm = confusion_matrix(test_mal_y, test_mal_adv_yh, labels=range(5))
	
	# update history
	grid_history['test_ben_acc'].append(test_ben_acc)
	grid_history['test_mal_acc'].append(test_mal_acc)
	grid_history['test_ben_adv_acc'].append(test_ben_adv_acc)
	grid_history['test_mal_adv_acc'].append(test_mal_adv_acc)
	grid_history['test_mal_adv_acct'].append(test_mal_adv_acct)
	grid_history['test_ben_cfm'].append(test_ben_cfm)
	grid_history['test_mal_cfm'].append(test_mal_cfm)
	grid_history['test_ben_adv_cfm'].append(test_ben_adv_cfm)
	grid_history['test_mal_adv_cfm'].append(test_mal_adv_cfm)
	
	# log progress
	if VERBOSE:
		print(f'Tree depth: {depth}')
		print('----')
		print(f'Baseline benign accuracy: {test_ben_acc}')
		print(f'Baseline malicious accuracy: {test_mal_acc}')
		print('----')
		print(f'Adversarial benign accuracy: {test_ben_adv_acc}')
		print(f'Adversarial malicious accuracy: {test_mal_adv_acc}')
		print(f'Adversarial malicious targeted accuracy: {test_mal_adv_acct}')
		print('----')
		print(test_ben_cfm)
		print(test_mal_cfm)
		print('----')
		print(test_ben_adv_cfm)
		print(test_mal_adv_cfm)


### plot results

# isolate scores
test_ben_acc = np.array(grid_history['test_ben_acc'])
test_mal_acc = np.array(grid_history['test_mal_acc'])
test_ben_adv_acc = np.array(grid_history['test_ben_adv_acc'])
test_mal_adv_acc = np.array(grid_history['test_mal_adv_acc'])
test_mal_adv_acct = np.array(grid_history['test_mal_adv_acct'])

# plots scores
fig, axs = plt.subplots(figsize=(6,4))
axs.plot(depth_axis, test_ben_adv_acc-test_ben_acc, label='benign')
axs.plot(depth_axis, test_mal_adv_acc-test_mal_acc, label='malicious', c='red')
axs.plot(depth_axis, test_mal_adv_acct-0, label='malicious targeted', linestyle='dashed', c='red')
axs.set_xlabel('Tree depth')
axs.set_ylabel('Delta-accuracy')
axs.grid()
axs.legend()
plt.savefig('decisiontree_ids_sklearn_dta_evaluate_depth.png')
