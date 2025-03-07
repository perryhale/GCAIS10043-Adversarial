import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import random
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers, callbacks
from imblearn.under_sampling import RandomUnderSampler

from library.random import split_key, seed_everything
from library.data import get_car_hacking_dataset, mask_fn
from library.models import get_multiclass_mlp, classifier_evaluation
from library.training import federated_uniform_adversarial_train
from library.attacks import BenMalPGD, SaltAndPepperNoise, benmalpgd_classifier_evaluation, spn_classifier_evaluation


### setup

# start timer
T0 = time.time()
print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# init RNG seeds
K0 = 999
K0, K1 = split_key(K0) # shuffle data
K1, K2 = split_key(K1) # init model
K2, K3 = split_key(K2) # global seed set at train time (workaround)


### hyperparameters

# data
FEATURES_RES = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255]).astype('float32')

# architecture
FEATURES_DIM = 10
LABELS_DIM = 5
HIDDEN_DIM = 16
HIDDEN_DEPTH = 4
HIDDEN_ACT = 'relu' # must be str for name formatting

# n_nodes gridsearch axis
###! uses range
NODE_MIN = 1
NODE_MAX = 9
NODE_STEP = 1 # {1, 2, ..., 8}

# max_strength gridsearch axis
###! uses linspace
MS_MIN = 1e-16
MS_MAX = 0.5
MS_DOF = 8

# training
N_EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.01
UNIF_LOW = 0.5
UNIF_HIGH = 1.0

# adversarial evaluation
EPS_MIN = 1e-16
EPS_MAX = 1.0
EPS_RES = 8
PGD_ITER = 7
SPN_MAG = 1.0

# tracing
VERBOSE = False


### prepare data

# load partitions
(train_x, train_y), (val_x, val_y), (test_x, test_y) = get_car_hacking_dataset(K1)

# undersample train set
###! train_x must be re-shuffled prior to node partitioning because RUS returns values sorted by class
train_x, train_y = RandomUnderSampler(random_state=K1).fit_resample(train_x, train_y)
train_indices = np.random.RandomState(seed=K1).permutation(len(train_x))
train_x = train_x[train_indices]
train_y = train_y[train_indices]

# generate masks from features
train_mask = np.apply_along_axis(mask_fn, axis=1, arr=train_x)
val_mask = np.apply_along_axis(mask_fn, axis=1, arr=val_x)
test_mask = np.apply_along_axis(mask_fn, axis=1, arr=test_x)

# normalise features
train_x = train_x / FEATURES_RES
val_x = val_x / FEATURES_RES
test_x = test_x / FEATURES_RES

# trace
print(train_x.shape, train_y.shape, 'train')
print(val_x.shape, val_y.shape, 'val')
print(test_x.shape, test_y.shape, 'test')
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### train and evaluate models

# initialise common lambdas
model_init = lambda:get_multiclass_mlp(
	K2,
	FEATURES_DIM,
	LABELS_DIM,
	HIDDEN_DIM,
	HIDDEN_DEPTH,
	hidden_act=HIDDEN_ACT
)
criterion_init = lambda:losses.SparseCategoricalCrossentropy()
optimizer_init = lambda:optimizers.AdamW(learning_rate=LEARNING_RATE)
metrics_init = lambda:['accuracy']

# init gridsearch
node_space = range(NODE_MIN, NODE_MAX, NODE_STEP)
ms_space = np.linspace(MS_MIN, MS_MAX, num=MS_DOF)
history = np.empty((len(node_space), len(ms_space)), dtype=object)

# trace
print(node_space)
print(ms_space)
print(history.shape)
print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# run gridsearch
for i, n_nodes in enumerate(node_space):
	for j, max_strength in enumerate(ms_space):
		
		###! set global RNG seeds
		seed_everything(K3)
		
		# initialise attack lambda
		attack_init = lambda model : BenMalPGD(
			model,
			FEATURES_DIM,
			LABELS_DIM,
			losses.SparseCategoricalCrossentropy(),
			epsilon=max_strength,
			iterations=PGD_ITER,
			batch_size=BATCH_SIZE,
			verbose=VERBOSE
		)
		
		# call train function
		train_history, model = federated_uniform_adversarial_train(
			model_init,
			criterion_init,
			optimizer_init,
			metrics_init,
			attack_init,
			FEATURES_RES,
			train_x,
			train_y,
			train_mask,
			val_x,
			val_y,
			val_mask,
			n_nodes=n_nodes,
			unif_lower=UNIF_LOW,
			unif_upper=UNIF_HIGH,
			epochs=N_EPOCHS,
			batch_size=BATCH_SIZE,
			callbacks=None,
			verbose=VERBOSE
		)
		model.name = f'BIDS_{HIDDEN_DIM}x{HIDDEN_DEPTH}_{HIDDEN_ACT}_Federated_N{n_nodes}_UPGDT_MS{max_strength:.2f}'.replace('.','_')
		model.summary()
		print(f'[Elapsed time: {time.time()-T0:.2f}s]')
		
		# create concrete objects
		criterion = criterion_init()
		
		# evaluate model
		test_history = classifier_evaluation(
			model,
			criterion,
			test_x,
			test_y,
			batch_size=BATCH_SIZE,
			verbose=VERBOSE
		)
		pgd_history = benmalpgd_classifier_evaluation(
			model,
			FEATURES_DIM,
			LABELS_DIM,
			criterion,
			test_x,
			test_y,
			mask=test_mask,
			feature_res=FEATURES_RES,
			eps_min=EPS_MIN,
			eps_max=EPS_MAX,
			eps_num=EPS_RES,
			pgd_iter=PGD_ITER,
			batch_size=BATCH_SIZE,
			verbose=VERBOSE
		)
		spn_history = spn_classifier_evaluation(
			model,
			criterion,
			test_x,
			test_y,
			mask=test_mask,
			feature_res=FEATURES_RES,
			eps_min=EPS_MIN,
			eps_max=EPS_MAX,
			eps_num=EPS_RES,
			spn_magnitude=SPN_MAG,
			batch_size=BATCH_SIZE,
			verbose=VERBOSE
		)
		
		# trace
		print(f'[Elapsed time: {time.time()-T0:.2f}s]')
		
		# checkpoint progress
		history[i][j] = {
			'info':{
				'name':model.name,
				'n_nodes':n_nodes,
				'max_strength':max_strength
			},
			'train':train_history,
			'test':test_history,
			'pgd':pgd_history,
			'spn':spn_history
		}
		with open(f'{__file__.replace(".py","")}_history.pkl', 'wb') as f: pickle.dump(history, f)
		model.save_weights(f'{model.name}.weights.h5')
		
		# trace
		print(train_history)
		print(test_history)
		print(pgd_history)
		print(spn_history)
		print(f'[Elapsed time: {time.time()-T0:.2f}s]')
