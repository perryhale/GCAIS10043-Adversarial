import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers, callbacks
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, MinMaxScaler
import pickle

from library.random import split_key
from library.data import get_car_hacking_dataset
from library.models import get_multiclass_mlp, classifier_evaluation
from library.attacks import BenMalPGD, SaltAndPepperNoise, benmalpgd_classifier_evaluation, spn_classifier_evaluation


### setup

# timer
T0 = time.time()
print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# RNG seed
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
HIDDEN_DEPTH = 5
HIDDEN_ACT = 'relu'

# training
N_EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.01

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

# standardize features
train_x_std = scale(train_x)
val_x_std = scale(val_x)
test_x_std = scale(test_x)

# fit data transform objects
data_pca = PCA().fit(train_x_std)
train_x_pct = data_pca.transform(train_x_std)
data_mms = MinMaxScaler().fit(train_x_pct)

# transform and normalise data
train_x_pctnorm = data_mms.transform(train_x_pct)
val_x_pctnorm = data_mms.transform(data_pca.transform(val_x_std))
test_x_pctnorm = data_mms.transform(data_pca.transform(test_x_std))

# trace
print(train_x.shape, train_y.shape, 'train')
print(val_x.shape, val_y.shape, 'val')
print(test_x.shape, test_y.shape, 'test')
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### train and evaluate models

# init gridsearch
nc_space = range(1, FEATURES_DIM+1)
hd_space = range(HIDDEN_DEPTH)
history = np.empty((len(nc_space), len(hd_space)), dtype=object)

# run gridsearch over architectural hyperparameters
for i, n_components in enumerate(nc_space):
	for j, hidden_depth in enumerate(hd_space):
		
		# init model
		criterion = losses.SparseCategoricalCrossentropy()
		optimizer = optimizers.AdamW(learning_rate=LEARNING_RATE)
		model = get_multiclass_mlp(
			K2,
			n_components,
			LABELS_DIM,
			HIDDEN_DIM,
			hidden_depth,
			hidden_act=HIDDEN_ACT,
			name=f'BIDS_{HIDDEN_DIM}x{hidden_depth}_{HIDDEN_ACT}_PC{n_components}'
		)
		model.compile(loss=criterion, optimizer=optimizer, metrics=['accuracy'])
		model.summary()
		
		###! set global RNG seeds
		###! prior to training
		np.random.seed(K3)
		tf.random.set_seed(K3)
		
		# train model
		train_history = model.fit(
			train_x_pctnorm[:,:n_components],
			train_y,
			epochs=N_EPOCHS,
			batch_size=BATCH_SIZE,
			validation_data=(val_x_pctnorm[:,:n_components], val_y),
			callbacks=None,
			verbose=int(VERBOSE)
		).history
		
		# trace
		print(f'[Elapsed time: {time.time()-T0:.2f}s]')
		
		# evaluate model
		test_history = classifier_evaluation(
			model,
			criterion,
			test_x_pctnorm[:,:n_components],
			test_y,
			batch_size=BATCH_SIZE,
			verbose=VERBOSE
		)
		pgd_history = benmalpgd_classifier_evaluation(
			model,
			FEATURES_DIM,
			LABELS_DIM,
			criterion,
			test_x_pctnorm[:,:n_components],
			test_y,
			mask=None, ###! no mask
			feature_res=None, ###! no enforced res
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
			test_x_pctnorm[:,:n_components],
			test_y,
			mask=None, ###! no mask
			feature_res=None, ###! no enforced res
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
				'n_components':n_components,
				'hidden_depth':hidden_depth
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
