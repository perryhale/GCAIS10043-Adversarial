import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers, callbacks
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm
import pickle

from library.random import split_key
from library.data import enforce_res, mask_fn, get_car_hacking_dataset
from library.models import get_multiclass_mlp
from library.training import scheduled_adversarial_train
from library.attacks import SaltAndPepperNoise, BenMalPGD


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

# gridsearch
MS_MIN = 1e-16
MS_MAX = 0.5
MS_RES = 8

# training
LEARNING_RATE = 0.001
L2_LAMBDA = 0.001
NUM_EPOCHS = 5
BATCH_SIZE = 512 ###! (512) decrease for truncation for debug and testing

# evaluation
PGD_ITER = 7
PGD_MIN = MS_MIN
PGD_MAX = MS_MAX
PGD_RES = 8

SPN_MAG = 1.0
SPN_MIN = 0.
SPN_MAX = PGD_MAX
SPN_RES = PGD_RES

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


### run gridsearch

# init history
history = {}
keys = split_key(K3, n=MS_RES)
for k3s, max_strength in zip(keys, np.linspace(MS_MIN, MS_MAX, num=MS_RES)):
	
	# init model
	model = get_multiclass_mlp(
		K2,
		FEATURES_DIM,
		LABELS_DIM,
		HIDDEN_DIM,
		HIDDEN_DEPTH,
		hidden_act=HIDDEN_ACT,
		l2_lambda=L2_LAMBDA,
		name=f'BIDS_{HIDDEN_DIM}x{HIDDEN_DEPTH}_{HIDDEN_ACT}_SPGDT_MS{max_strength:.2f}'.replace('.','_')
	)
	
	# compile model
	criterion = tf.keras.losses.SparseCategoricalCrossentropy()
	optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE)
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
	np.random.seed(k3s)
	tf.random.set_seed(k3s)
	
	# define callbacks
	checkpoint_callback = callbacks.ModelCheckpoint(
		filepath=f'{model.name}.weights.h5',
		monitor='val_loss',
		mode='max',
		save_weights_only=True,
		save_best_only=False
	)
	
	# call train function
	train_history = None
	try:
		train_history = scheduled_adversarial_train(
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
			callbacks=[checkpoint_callback],
			verbose=VERBOSE
		)
	
	# trace
	finally:
		print(f'[Elapsed time: {time.time()-T0:.2f}s]')
	
	# reload if save_best_only=True
	###!
	
	# compute baseline scores
	test_yh = model.predict(test_x, batch_size=BATCH_SIZE, verbose=int(VERBOSE))
	test_loss = criterion(test_y, test_yh).numpy()
	test_acc = accuracy_score(test_y, np.argmax(test_yh, axis=-1))
	test_cfm = confusion_matrix(test_y, np.argmax(test_yh, axis=-1), labels=range(LABELS_DIM))
	test_history = dict(
		loss=test_loss,
		accuracy=test_acc,
		confusion=test_cfm
	)
	
	# evaluate pgd adversary
	pgd_history = {
		'epsilon':[],
		'loss':[],
		'accuracy':[],
		'confusion':[]
	}
	for eps in tqdm(np.linspace(PGD_MIN, PGD_MAX, num=PGD_RES), desc='PGD', unit='epsilon'):
		
		# generate samples
		attack = BenMalPGD(model, FEATURES_DIM, LABELS_DIM, criterion, epsilon=eps, iterations=PGD_ITER, batch_size=BATCH_SIZE, verbose=VERBOSE)
		test_adv_x = enforce_res(attack.generate(test_x, test_y, mask=test_mask), FEATURES_RES)
		
		# evaluate model
		test_adv_yh = model.predict(test_adv_x, batch_size=BATCH_SIZE, verbose=int(VERBOSE))
		test_adv_loss = criterion(test_y, test_adv_yh).numpy()
		test_adv_acc = accuracy_score(test_y, np.argmax(test_adv_yh, axis=-1))
		test_adv_cfm = confusion_matrix(test_y, np.argmax(test_adv_yh, axis=-1), labels=range(LABELS_DIM))
		
		# record results
		pgd_history['epsilon'].append(eps)
		pgd_history['loss'].append(test_adv_loss)
		pgd_history['accuracy'].append(test_adv_acc)
		pgd_history['confusion'].append(test_adv_cfm)
	
	# evaluate spn adversary
	spn_history = {
		'ratio':[],
		'loss':[],
		'accuracy':[],
		'confusion':[]
	}
	for ratio in tqdm(np.linspace(PGD_MIN, PGD_MAX, num=PGD_RES), desc='SPN', unit='noise_ratio'):
		
		# generate samples
		attack = SaltAndPepperNoise(noise_ratio=ratio, noise_magnitude=1.0)
		test_adv_x = enforce_res(attack.generate(test_x, test_y, mask=test_mask), FEATURES_RES)
		
		# evaluate model
		test_adv_yh = model.predict(test_adv_x, batch_size=BATCH_SIZE, verbose=int(VERBOSE))
		test_adv_loss = criterion(test_y, test_adv_yh).numpy()
		test_adv_acc = accuracy_score(test_y, np.argmax(test_adv_yh, axis=-1))
		test_adv_cfm = confusion_matrix(test_y, np.argmax(test_adv_yh, axis=-1), labels=range(LABELS_DIM))
		
		# record results
		spn_history['ratio'].append(ratio)
		spn_history['loss'].append(test_adv_loss)
		spn_history['accuracy'].append(test_adv_acc)
		spn_history['confusion'].append(test_adv_cfm)
	
	# update history
	history.update({
		model.name:{
			'max_strength':max_strength,
			'train':train_history,
			'test':test_history,
			'pgd':pgd_history,
			'spn':spn_history
		}
	})
	print(history)
	
	# checkpoint history
	with open('adversarial_train_pgd_scheduled_history.pkl', 'wb') as f:
		pickle.dump(history, f)
