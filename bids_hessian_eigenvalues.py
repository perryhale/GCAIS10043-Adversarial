import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# model
import tensorflow as tf
from tensorflow.keras import (
	layers,
	initializers,
	regularizers,
	optimizers,
	callbacks
)
from sklearn.metrics import confusion_matrix, accuracy_score

# adversary
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import ProjectedGradientDescent

# util
import time
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


### start timer

T0 = time.time()


### init RNG seeds

# type: (int, int) -> List[int]
def split_key(key, n=2):
	return [key * (i+1) for i in range(n)]

K0 = 999
K0, K1 = split_key(K0) # shuffle data


### functions

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
	data = data.sample(frac=1, random_state=key)[:100_000] ###! truncation must be 100_000 (29/01/25) to match train truncation
	
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

# type: (int, int, int, int, int, str, float, str) -> tf.keras.Model
def get_multiclass_mlp(
		key,
		input_dim,
		output_dim,
		hidden_dim,
		hidden_depth,
		hidden_act='relu',
		l2_lambda=0.0,
		name='Multiclass-MLP'
	):
	
	# split key
	keys = split_key(key, n=hidden_depth+1)
	
	# init input
	model_x = layers.Input(shape=(input_dim,), name='input')
	model_y = model_x
	
	# init hidden
	for i in range(hidden_depth):
		model_y = layers.Dense(
			hidden_dim,
			activation=hidden_act,
			kernel_initializer=initializers.GlorotUniform(seed=keys[i]),
			kernel_regularizer=regularizers.l2(l2_lambda),
			name=f'hidden{i+1}'
		)(model_y)
	
	# init output
	model_y = layers.Dense(
		output_dim,
		activation='softmax',
		kernel_initializer=initializers.GlorotUniform(seed=keys[-1]),
		kernel_regularizer=regularizers.l2(l2_lambda),
		name='output'
	)(model_y)
	
	model = tf.keras.Model(model_x, model_y, name=name)
	return model

# type: (np.ndarray, np.ndarray, tf.keras.Model, tf.keras.Loss, float) -> float
def compute_hessian_eigenvalues(x, y, model, loss_fn, batch_size=32, verbose=False):
	
	# initialise
	x_batches = np.array_split(x, len(x) // batch_size)
	yh = []
	hessian = []
	
	# start gradient tape
	with tf.GradientTape(persistent=True) as tape:
		
		# compute batched predictions
		for batch in x_batches:
			batch_yh = model(batch)
			yh.append(batch_yh)
		yh = tf.concat(yh, axis=0)
		
		# compute first order gradients
		loss = loss_fn(y, yh)
		gradients = tape.gradient(loss, model.trainable_variables)
		gradients = tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0)
		
		# compute second order gradients
		for g in tqdm(gradients, desc='Compute hessian') if verbose else gradients:
			hessian_row = tape.gradient(g, model.trainable_variables)
			hessian_row = tf.concat([tf.reshape(h, [-1]) for h in hessian_row], axis=0)
			hessian.append(hessian_row)
	
	# finalise
	hessian = tf.stack(hessian, axis=0)
	hessian_eigenvalues = tf.linalg.eigvalsh(hessian)
	
	return hessian_eigenvalues


### hyperparameters

# data
FEATURES_RES = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255]).astype('float32')
BATCH_SIZE = 512

# architecture
FEATURES_DIM = 10
LABELS_DIM = 5
HIDDEN_DIM = 16
HIDDEN_DEPTH = 4
HIDDEN_ACT = 'relu' # must be str for name formatting

# tracing
VERBOSE = False


### evaluate models

# choose postfix
options = {'-u':'_uniform', '-s':'_scheduled'}
postfix = options[sys.argv[1]] if (len(sys.argv) > 1 and sys.argv[1] in options.keys()) else '_uniform'

# load dictionary
with open(f'bids_advpgd{postfix}_history.pkl', 'rb') as f:
	history = pickle.load(f)

# load dataset
_, _, (test_x, test_y) = get_car_hacking_dataset(K1)
test_x = test_x / FEATURES_RES
print(test_x.shape, test_y.shape, 'test')

# run search
for name_key in tqdm(history.keys(), desc='HEVs', unit='model'):
	
	# init model
	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
	model = get_multiclass_mlp(
		0, # degen key
		FEATURES_DIM,
		LABELS_DIM,
		HIDDEN_DIM,
		HIDDEN_DEPTH,
		hidden_act=HIDDEN_ACT,
		name=name_key
	)
	model.load_weights(f'{name_key}.weights.h5')
	
	# compute hessian eigenvalues
	hevs = compute_hessian_eigenvalues(test_x, test_y, model, loss_fn, batch_size=BATCH_SIZE, verbose=VERBOSE)
	
	# record results
	history[name_key]['test'].update({'hevs':hevs})
	if VERBOSE:
		print(name_key)
		print(hevs)
		print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# save history
with open(f'bids_advpgd{postfix}_history_hevs.pkl', 'wb') as f:
	print(history)
	pickle.dump(history, f)
