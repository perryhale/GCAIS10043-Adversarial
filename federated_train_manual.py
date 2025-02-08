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
K1, K2 = split_key(K1) # init model
K2, K3 = split_key(K2) # global seed set at train time (workaround)


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

# type: (List[tf.keras.Model], List[float], str) -> tf.keras.Model
###! assumes all models are Multiclass-MLPs of the same shape
def merge_multiclass_mlps(models, weights=None, name='Merged Multiclass-MLP'):
	
	# weights setup
	if weights is None:
		weights = np.ones((len(models))) / len(models)
	else:
		msg = f"The number of weights provided must match the number of models. got <{len(models)}> <{len(weights)}>"
		assert len(models)==len(weights), msg
	
	# zero-initialise merged model
	merged_model = tf.keras.models.clone_model(models[0])
	merged_model.name = name
	for layer in merged_model.layers:
		layer.set_weights([np.zeros_like(w) for w in layer.get_weights()])
	
	# accumulate scaled parameters
	for model, weight in zip(models, weights):
		for layer, merge_layer in zip(model.layers, merged_model.layers):
			merge_layer.set_weights([mw+w*weight for w,mw in zip(layer.get_weights(), merge_layer.get_weights())])
	
	return merged_model

# type: () ->
###! use global RNG to control randomness
"""def federated_train(
		model,
		optmizer_params,
		train_params,
		train_params,
		verbose=False
	):
	
	# unpack params
	criterion, optimizer, metrics = optmizer_params
	train_x, train_y, val_x, val_y = data_params
	n_nodes, epochs, batch_size, callbacks = train_params # packed params"""
def federated_train(
		model,
		criterion,
		optimizer,
		metrics,
		train_x,
		train_y,
		val_x,
		val_y,
		n_nodes=5,
		epochs=10,
		batch_size=64,
		callbacks=None,
		verbose=False
	):
	
	# initialise history
	history = {'nodes':[], 'weights':[]}
	
	# split data
	train_x_partitions = np.array_split(train_x, n_nodes)
	train_y_partitions = np.array_split(train_y, n_nodes)
	
	# initialise nodes
	nodes = []
	for i in range(n_nodes):
		node = model()
		node.name = f'federated_node_{i+1}'
		node.compile(loss=criterion(), optimizer=optimizer(), metrics=metrics)
		nodes.append(node)
	
	node_weights = [len(txp)/len(train_x) for txp in train_x_partitions]
	history['weights'] = node_weights
	
	# train nodes
	#for node, train_x_partition, train_y_partition in zip(nodes, train_x_partitions, train_y_partitions): # no tqdm
	train_zip = list(zip(nodes, train_x_partitions, train_y_partitions))
	for i in tqdm(range(len(train_zip)), desc='Federated train', unit='node'):
		node, train_x_partition, train_y_partition = train_zip[i] # tqdm workaround for progress bar
		node_history = node.fit(
			train_x_partition,
			train_y_partition,
			epochs=epochs,
			batch_size=batch_size,
			validation_data=(val_x, val_y),
			callbacks=callbacks,
			verbose=int(verbose)
		)
		history['nodes'].append(node_history.history)
	
	# merge nodes
	federated_model = merge_multiclass_mlps(nodes, weights=node_weights)
	
	return history, federated_model


### hyperparameters

# data
FEATURES_RES = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255]).astype('float32')

# architecture
FEATURES_DIM = 10
LABELS_DIM = 5
HIDDEN_DIM = 16
HIDDEN_DEPTH = 4
HIDDEN_ACT = 'relu' # must be str for name formatting

# training
N_NODES = 4
N_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.01

# tracing
VERBOSE = False


### prepare data

# load partitions
(train_x, train_y), (val_x, val_y), (test_x, test_y) = get_car_hacking_dataset(K1)

# undersample train set
###! train_x must be re-shuffled prior to node partitioning because RUS returns sorted values
train_x, train_y = RandomUnderSampler(random_state=K1).fit_resample(train_x, train_y)
train_indices = np.random.RandomState(seed=K1).permutation(len(train_x))
train_x = train_x[train_indices]
train_y = train_y[train_indices]

# normalise features
train_x = train_x / FEATURES_RES
val_x = val_x / FEATURES_RES
test_x = test_x / FEATURES_RES

# trace
print(train_x.shape, train_y.shape, 'train')
print(val_x.shape, val_y.shape, 'val')
print(test_x.shape, test_y.shape, 'test')
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### train model

# initialise
###! model not precompiled because each node must be compiled again later
###! criterion and optimizer must be lambdas in order to correctly compile nodes
###! model is lambda to allow arbitrary cloning implementation
###! metrics would also need to be lambda if using stateful metrics
model = lambda:get_multiclass_mlp(
	K2,
	FEATURES_DIM,
	LABELS_DIM,
	HIDDEN_DIM,
	HIDDEN_DEPTH,
	hidden_act=HIDDEN_ACT
)
criterion = lambda:tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = lambda:tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE)
metrics = ['accuracy']

###! set global RNG seeds
###! prior to training
np.random.seed(K3)
tf.random.set_seed(K3)

# call train function
train_history, model = federated_train(
	model,
	criterion,
	optimizer,
	metrics,
	train_x,
	train_y,
	val_x,
	val_y,
	n_nodes=N_NODES,
	epochs=N_EPOCHS,
	batch_size=BATCH_SIZE,
	callbacks=None,
	verbose=VERBOSE
)
model.name = f'BIDS_{HIDDEN_DIM}x{HIDDEN_DEPTH}_{HIDDEN_ACT}_Federated_N{N_NODES}'
model.summary()


### evaluate model

test_yh = model.predict(test_x, batch_size=BATCH_SIZE, verbose=int(VERBOSE))
test_loss = criterion()(test_y, test_yh).numpy()
test_acc = accuracy_score(test_y, np.argmax(test_yh, axis=-1))
test_cfm = confusion_matrix(test_y, np.argmax(test_yh, axis=-1), labels=range(LABELS_DIM))
test_history = dict(
	loss=test_loss,
	accuracy=test_acc,
	confusion=test_cfm
)
print(train_history)
print(test_history)
