import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers, callbacks
from imblearn.under_sampling import RandomUnderSampler
import pickle

from library.random import split_key
from library.data import get_car_hacking_dataset, mask_fn
from library.models import get_multiclass_mlp, classifier_evaluation
from library.training import uniform_adversarial_train
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

# gridsearch
MS_MIN = 1e-16
MS_MAX = 0.5
MS_RES = 8

# training
LEARNING_RATE = 0.001
L2_LAMBDA = 0.001
NUM_EPOCHS = 5
BATCH_SIZE = 256

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
train_x, train_y = RandomUnderSampler(random_state=K1).fit_resample(train_x, train_y)

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

# init gridsearch
ms_space = np.linspace(MS_MIN, MS_MAX, num=MS_RES)
history = np.empty(len(ms_space), dtype=object)

# trace
print(ms_space)
print(history.shape)
print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# run gridsearch
for i, max_strength in enumerate(ms_space):
	
	# init model
	criterion = tf.keras.losses.SparseCategoricalCrossentropy()
	optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE)
	model = get_multiclass_mlp(
		K2,
		FEATURES_DIM,
		LABELS_DIM,
		HIDDEN_DIM,
		HIDDEN_DEPTH,
		hidden_act=HIDDEN_ACT,
		l2_lambda=L2_LAMBDA,
		name=f'BIDS_{HIDDEN_DIM}x{HIDDEN_DEPTH}_{HIDDEN_ACT}_UPGDT_MS{max_strength:.2f}'.replace('.','_')
	)
	model.compile(loss=criterion, optimizer=optimizer, metrics=['accuracy'])
	
	# init attack
	attack = BenMalPGD(
		model,
		FEATURES_DIM,
		LABELS_DIM,
		criterion,
		epsilon=max_strength,
		iterations=PGD_ITER,
		batch_size=BATCH_SIZE,
		verbose=VERBOSE
	)
	
	# trace
	model.summary()
	print(f'[Elapsed time: {time.time()-T0:.2f}s]')
	
	###! set global RNG seeds
	###! prior to training
	np.random.seed(K3)
	tf.random.set_seed(K3)
	
	# call train function
	train_history = uniform_adversarial_train(
		model,
		attack,
		FEATURES_RES,
		train_x,
		train_y,
		train_mask,
		val_x,
		val_y,
		val_mask,
		epochs=NUM_EPOCHS,
		batch_size=BATCH_SIZE,
		callbacks=None,
		verbose=VERBOSE
	)
	
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
	history[i] = {
		'info':{
			'name':model.name,
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
