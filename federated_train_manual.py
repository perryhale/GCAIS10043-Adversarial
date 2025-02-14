import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers, callbacks
from imblearn.under_sampling import RandomUnderSampler

from library.random import split_key
from library.data import get_car_hacking_dataset
from library.models import get_multiclass_mlp, classifier_evaluation
from library.training import federated_train


### setup

# start timer
T0 = time.time()

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
###! train_x must be re-shuffled prior to node partitioning because RUS returns values sorted by class
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


### train and evaluate model

# initialise
###! model not precompiled because each node must be compiled again later
###! criterion and optimizer must be lambdas in order to correctly compile nodes
###! model is lambda to allow arbitrary cloning implementation
###! metrics would also need to be lambda if using stateful metrics
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

###! set global RNG seeds
###! prior to training
np.random.seed(K3)
tf.random.set_seed(K3)

# call train function
train_history, model = federated_train(
	model_init,
	criterion_init,
	optimizer_init,
	metrics_init,
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
print(train_history)
print(test_history)
print(f'[Elapsed time: {time.time()-T0:.2f}s]')
