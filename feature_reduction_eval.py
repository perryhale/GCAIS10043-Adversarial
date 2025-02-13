"""

###! attack functions are incompatible with reduced dimensions 


"""


import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers, callbacks
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, MinMaxScaler
from tqdm import tqdm
import pickle

from library.random import split_key
from library.data import get_car_hacking_dataset
from library.models import get_multiclass_mlp
from library.attacks import BenMalPGD, SaltAndPepperNoise, benmalpgd_evaluation, spn_evaluation


### hyperparameters

# timer
T0 = time.time()

# RNG seed
K0 = 999
K0, K1 = split_key(K0) # shuffle data
K1, K2 = split_key(K1) # init model
K2, K3 = split_key(K2) # global seed set at train time (workaround)

# data
FEATURES_RES = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255]).astype('float32')

# architecture
FEATURES_DIM = 10
CLASSES_DIM = 5
HIDDEN_DIM = 16
HIDDEN_DEPTH = 5
HIDDEN_ACT = 'relu'

# training
N_EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.01

# pgd evaluation
PGD_MIN = 1e-16
PGD_MAX = 1.0
PGD_RES = 8
PGD_ITER = 7

# spn evaluation
SPN_MIN = PGD_MIN
SPN_MAX = PGD_MAX
SPN_RES = PGD_RES
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

# trace
print(train_x.shape, train_y.shape, 'train')
print(val_x.shape, val_y.shape, 'val')
print(test_x.shape, test_y.shape, 'test')
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### train and evaluate models

# run gridsearch over architectural hyperparameters
for n_components in range(1, FEATURES_DIM+1):
	for hidden_depth in range(HIDDEN_DEPTH):
		
		# data: fit transform objects
		data_pca = PCA(n_components=n_components).fit(train_x_std)
		train_x_pct = data_pca.transform(train_x_std)
		data_mms = MinMaxScaler().fit(train_x_pct)
		
		# data: transform and normalise
		train_x_pctnorm = data_mms.transform(train_x_pct)
		val_x_pctnorm = data_mms.transform(data_pca.transform(val_x_std))
		test_x_pctnorm = data_mms.transform(data_pca.transform(test_x_std))
		
		# init model
		model = get_multiclass_mlp(
			K2,
			n_components,
			CLASSES_DIM,
			HIDDEN_DIM,
			hidden_depth,
			hidden_act=HIDDEN_ACT,
			name=f'BIDS_{HIDDEN_DIM}x{hidden_depth}_{HIDDEN_ACT}_FR{n_components}'
		)
		criterion = losses.SparseCategoricalCrossentropy()
		optimizer = optimizers.AdamW(learning_rate=LEARNING_RATE)
		model.compile(loss=criterion, optimizer=optimizer, metrics=['accuracy'])
		model.summary()
		
		# train model
		train_history = model.fit(
			train_x_pctnorm,
			train_y,
			epochs=N_EPOCHS,
			batch_size=BATCH_SIZE,
			validation_data=(val_x_pctnorm, val_y),
			callbacks=None,
			verbose=int(VERBOSE)
		)

		# evaluate baseline
		test_yh = model.predict(test_x_pctnorm, batch_size=BATCH_SIZE, verbose=int(VERBOSE))
		test_loss = criterion(test_y, test_yh).numpy().item()
		test_acc = accuracy_score(test_y, np.argmax(test_yh, axis=-1))
		test_cfm = confusion_matrix(test_y, np.argmax(test_yh, axis=-1), labels=range(CLASSES_DIM))
		test_history = dict(
			loss=test_loss,
			accuracy=test_acc,
			confusion=test_cfm
		)
		
		# evalute adversarial performance
		pgd_history = benmalpgd_evaluation(
			model,
			n_components,
			CLASSES_DIM,
			criterion,
			FEATURES_RES,
			test_x_pctnorm,
			test_y,
			np.ones_like(test_x_pctnorm),
			eps_min=PGD_MIN,
			eps_max=PGD_MAX,
			eps_num=PGD_RES,
			pgd_iter=PGD_ITER,
			batch_size=BATCH_SIZE,
			verbose=VERBOSE
		)
		spn_history = spn_evaluation(
			model,
			criterion,
			CLASSES_DIM,
			FEATURES_RES,
			test_x_pctnorm,
			test_y,
			np.ones_like(test_x_pctnorm),
			eps_min=SPN_MIN,
			eps_max=SPN_MAX,
			eps_num=SPN_RES,
			spn_magnitude=SPN_MAG,
			batch_size=BATCH_SIZE,
			verbose=VERBOSE
		)
		
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
			'spn':spn_history,
		}
		with open('federated_adversarial_train_pgd_uniform_history.pkl', 'wb') as f: pickle.dump(history, f)
		model.save_weights(f'{model.name}.weights.h5')
		
		# trace
		print(train_history)
		print(test_history)
		print(pgd_history)
		print(spn_history)
		print(f'[Elapsed time: {time.time()-T0:.2f}s]')
