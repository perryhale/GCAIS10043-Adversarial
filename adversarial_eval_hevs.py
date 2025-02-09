import os
import sys
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from tensorflow.keras import losses
from tqdm import tqdm
import pickle

from library.random import split_key
from library.data import get_car_hacking_dataset
from library.models import get_multiclass_mlp, compute_hessian


### setup

# choose postfix
OPTIONS = {'-u':'_uniform', '-s':'_scheduled'}
POSTFIX = OPTIONS[sys.argv[1]] if (len(sys.argv) > 1 and sys.argv[1] in OPTIONS.keys()) else '_uniform'

# start timer
T0 = time.time()

# init RNG seeds
K0 = 999
K0, K1 = split_key(K0) # shuffle data


### hyperparameters

# data
FEATURES_RES = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255]).astype('float32')
BATCH_SIZE = 128

# architecture
FEATURES_DIM = 10
LABELS_DIM = 5
HIDDEN_DIM = 16
HIDDEN_DEPTH = 4
HIDDEN_ACT = 'relu' # must be str for name formatting

# tracing
VERBOSE = False


### evaluate models

# reload dictionary
with open(f'adversarial_train_pgd{POSTFIX}_history.pkl', 'rb') as f:
	history = pickle.load(f)

# load dataset
_, _, (test_x, test_y) = get_car_hacking_dataset(K1)
test_x = test_x / FEATURES_RES
print(test_x.shape, test_y.shape, 'test')

# compute hevs
for name_key in tqdm(history.keys(), desc='HEVs', unit='model'):
	
	# init model
	loss_fn = losses.SparseCategoricalCrossentropy()
	model = get_multiclass_mlp(
		0,
		FEATURES_DIM,
		LABELS_DIM,
		HIDDEN_DIM,
		HIDDEN_DEPTH,
		hidden_act=HIDDEN_ACT,
		name=name_key
	)
	model.load_weights(f'{name_key}.weights.h5')
	
	# compute hessian eigenvalues
	hessian = compute_hessian(model, loss_fn, test_x, test_y, batch_size=BATCH_SIZE, verbose=VERBOSE)
	hevs = np.linalg.eigvals(hessian)
	
	# record results
	history[name_key]['test'].update({'hessian':hessian, 'hevs':hevs})
	if VERBOSE:
		print(name_key)
		print(hevs)
		print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# save history
with open(f'adversarial_train_pgd{POSTFIX}_history_hevs.pkl', 'wb') as f:
	print(history)
	pickle.dump(history, f)
