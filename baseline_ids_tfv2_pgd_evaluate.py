import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers

import art
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import ProjectedGradientDescent

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics.pairwise import cosine_distances


### set RNG seed
RNG_SEED = 0
tf.keras.utils.set_random_seed(RNG_SEED)
np.random.seed(RNG_SEED)


### data setup
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

data = pd.read_csv('car_hacking_dataset/car_hacking_dataset.csv', header=None)
data = data.sample(frac=1)[:25_000]
train_data, val_data, test_data = train_val_test_split(data)
train_x, train_y = standard_split(train_data)
val_x, val_y = standard_split(val_data)
(test_ben_x, test_ben_y, test_ben_mask), (test_mal_x, test_mal_y, test_mal_yt, test_mal_mask) = adversarial_split(test_data)

print(train_x.shape, train_y.shape, 'train')
print(val_x.shape, val_y.shape, 'validation')
print(test_ben_x.shape, test_ben_y.shape, 'test (benign)')
print(test_mal_x.shape, test_mal_y.shape, 'test (malicious)')


### evaluate models
MODEL_PATH = 'models/'
MODELS = ['baseline_ids_tfv2_pgd_train', 'baseline_ids_tfv2_pgd_train_us', 'baseline_ids_tfv2_pgd_train_us_i5', 'baseline_ids_tfv2_pgd_train_us_e5']
EPS_RES = 32
EPS_MIN = 1e-9
EPS_MAX = 1.0
PGD_ITER = 7
VERBOSE = True

model_history = {k:{} for k in MODELS}

for k in MODELS:
	
	# define model
	model_x = layers.Input(shape=(10,), name='input')
	model_y = layers.Dense(16, activation='relu', name='hidden1')(model_x)
	model_y = layers.Dense(16, activation='relu', name='hidden2')(model_y)
	model_y = layers.Dense(16, activation='relu', name='hidden3')(model_y)
	model_y = layers.Dense(16, activation='relu', name='hidden4')(model_y)
	model_y = layers.Dense(5, activation='softmax', name='output')(model_y)
	model = tf.keras.Model(model_x, model_y, name=k)
	model.load_weights(MODEL_PATH+k+'.weights.h5')
	model.summary()
	
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
	
	for e in tqdm(np.linspace(EPS_MIN, EPS_MAX, num=EPS_RES)):
		
		# setup art wrapper
		art_model = TensorFlowV2Classifier(model=model, nb_classes=5, input_shape=(10,), loss_object=tf.keras.losses.SparseCategoricalCrossentropy(), clip_values=(0,1))
		pgd_untargeted = ProjectedGradientDescent(estimator=art_model, eps=e, eps_step=(e/PGD_ITER), max_iter=PGD_ITER, num_random_init=1, targeted=False, batch_size=8192, verbose=VERBOSE)
		pgd_targeted = ProjectedGradientDescent(estimator=art_model, eps=e, eps_step=(e/PGD_ITER), max_iter=PGD_ITER, num_random_init=1, targeted=True, batch_size=8192, verbose=VERBOSE)
		
		# generate samples
		test_ben_adv_x = enforce_res(pgd_untargeted.generate(test_ben_x, mask=test_ben_mask), FEATURE_SCALE, mask=test_ben_mask)
		test_mal_adv_x = enforce_res(pgd_targeted.generate(test_mal_x, test_mal_yt, mask=test_mal_mask), FEATURE_SCALE, mask=test_mal_mask)
		
		# evaluate model on samples
		test_ben_yh = np.argmax(model.predict(test_ben_x), axis=-1)
		test_mal_yh = np.argmax(model.predict(test_mal_x), axis=-1)
		test_ben_adv_yh = np.argmax(model.predict(test_ben_adv_x), axis=-1)
		test_mal_adv_yh = np.argmax(model.predict(test_mal_adv_x), axis=-1)
		
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
		model_history[k].update(grid_history)
		
		# log progress
		if VERBOSE:
			print('----')
			print(k)
			print(f'epsilon={e} iterations={PGD_ITER}')
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
x_axis = np.linspace(0, 1, num=EPS_RES)

fig, axs = plt.subplots(1, len(MODELS), figsize=(6*len(MODELS), 4))
for i,k in enumerate(MODELS):
	axs[i].set_title(k)
	axs[i].plot(x_axis, model_history[k]['test_ben_adv_acc'], label='benign')
	axs[i].plot(x_axis, model_history[k]['test_mal_adv_acc'], label='malicious', c='red')
	axs[i].plot(x_axis, model_history[k]['test_mal_adv_acct'], label='malicious targeted', linestyle='dashed', c='red')
	axs[i].set_xlabel('ε')
	if i==0:
		axs[i].set_ylabel('Accuracy')
	axs[i].grid()
	#axs[i].legend()

plt.show()
plt.clf()
