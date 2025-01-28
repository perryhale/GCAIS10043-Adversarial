import os
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

# type: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
def enforce_res(xs, res, mask=None):
	res = xs - np.round(np.minimum(res, np.maximum(0., xs * res))) / res
	if mask is not None:
		res *= mask
	return xs - res

# type: (np.ndarray) -> np.ndarray
def mask_fn(x):
	mask = np.zeros(x.shape)
	mask[2:2+x[1]] = 1.
	return mask

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
	data = data.sample(frac=1, random_state=key)[:100_000] ###! truncation for debug and testing
	
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

# type: () ->
###! no straightforward way to make ART deterministic
###! currently, should set global seed to control randomness
def scheduled_adversarial_train(
		attack,
		model,
		criterion,
		optimizer,
		input_dim,
		output_dim,
		feature_res,
		train_x,
		train_y,
		train_mask,
		val_x,
		val_y,
		val_mask,
		schedule_start=1.0,
		schedule_stop=0.0,
		epochs=10,
		batch_size=64,
		callbacks=None,
		verbose=False
	):
	
	# assertions
	assert model.optimizer is not None and model.loss is not None and model.metrics is not None, "model must be precompiled"
	
	# init history
	train_history = {
		
		# train set
		'loss':[],
		'accuracy':[],
		'adv_accuracy':[],
		
		# val set
		'val_loss':[],
		'val_accuracy':[],
		'val_adv_accuracy':[]
	}
	
	# train model
	with tqdm(np.linspace(schedule_start, schedule_stop, num=epochs), desc='Train', unit='epoch') as bar:
		for i in bar:
			
			# generate adversarial samples with scaled perturbations
			train_adv_xres = enforce_res(attack.generate(train_x, train_y, mask=train_mask), feature_res) - train_x ###! ND seed
			val_adv_xres = enforce_res(attack.generate(val_x, val_y, mask=val_mask), feature_res) - val_x ###! ND seed
			train_adv_x = train_x + train_adv_xres * i
			val_adv_x = val_x + val_adv_xres * i
			if verbose:
				print(f'[Elapsed time: {time.time()-T0:.2f}s]')
			
			# fit to augmented training set
			train_aug_x = np.concatenate([train_x, train_adv_x])
			train_aug_y = np.concatenate([train_y, train_y])
			epoch_hist = model.fit(
				train_aug_x,
				train_aug_y,
				epochs=1,
				batch_size=batch_size,
				validation_data=(val_x, val_y),
				callbacks=callbacks,
				verbose=int(verbose)
			) ###! ND seed
			
			# record results
			for k in epoch_hist.history.keys():
				train_history[k].extend(epoch_hist.history[k])
			if verbose:
				print(f'[Elapsed time: {time.time()-T0:.2f}s]')
			
			# evaluate adversarial accuracy
			train_adv_yh = model.predict(train_adv_x, batch_size=BATCH_SIZE, verbose=int(verbose))
			train_adv_acc = accuracy_score(train_y, np.argmax(train_adv_yh, axis=-1))
			val_adv_yh = model.predict(val_adv_x, batch_size=BATCH_SIZE, verbose=int(verbose))
			val_adv_acc = accuracy_score(val_y, np.argmax(val_adv_yh, axis=-1))
			
			# record results
			train_history['adv_accuracy'].append(train_adv_acc),
			train_history['val_adv_accuracy'].append(val_adv_acc)
			
			# trace
			if verbose:
				print(f'[Elapsed time: {time.time()-T0:.2f}s]')
				print(f'Epoch {i+1}: '+', '.join([f'{k}={train_history[k][-1]}' for k in train_history.keys()]))
			
			bar.set_postfix(
				loss=f'{train_history["loss"][-1]:.4f}',
				accuracy=f'{train_history["accuracy"][-1]:.4f}',
				val_loss=f'{train_history["val_loss"][-1]:.4f}',
				val_acc=f'{train_history["val_accuracy"][-1]:.4f}'
			)
	
	return train_history


### define custom attacks

class SaltAndPepperNoise():
	def __init__(self, noise_ratio=0.1, noise_magnitude=1.0):
		self.noise_ratio = noise_ratio
		self.noise_magnitude = noise_magnitude
	
	def generate(self, x, y, mask=None):
		###! redundant arg y, included for interface consistency
		noise = np.random.uniform(-1, 1, x.shape).astype(np.float32)
		noise = np.where(np.abs(noise) <= self.noise_ratio, self.noise_magnitude * (np.sign(noise) / 2 + 0.5) - x, 0)
		if mask is not None:
			noise *= mask
		return x + noise

# x = np.ones((32,32,1)) / 2
# plt.imshow(x)
# plt.colorbar()
# plt.show()
# xh = SaltAndPepperNoise().generate(x)
# plt.imshow(xh)
# plt.colorbar()
# plt.show()
# import sys;sys.exit()

class BenMalPGD():
	def __init__(self, model, input_dim, output_dim, criterion, epsilon=0.1, iterations=7, batch_size=32, verbose=False):
		art_model = TensorFlowV2Classifier(
			model=model,
			input_shape=(input_dim,),
			nb_classes=output_dim,
			loss_object=criterion,
			clip_values=(0,1)
		)
		self.pgd_untargeted = ProjectedGradientDescent(
			estimator=art_model,
			eps=epsilon,
			eps_step=epsilon/iterations,
			max_iter=iterations,
			num_random_init=1,
			targeted=False,
			batch_size=batch_size,
			verbose=verbose
		)
		self.pgd_targeted = ProjectedGradientDescent(
			estimator=art_model,
			eps=epsilon,
			eps_step=epsilon/iterations,
			max_iter=iterations,
			num_random_init=1,
			targeted=True,
			batch_size=batch_size,
			verbose=verbose
		)
	
	def generate(self, x, y, mask=None):
		
		# merge [x, y, mask] and split by class category
		data = pd.DataFrame(np.concatenate([x, y.reshape((y.shape[0], 1,)), mask], axis=-1))
		data_ben = data.loc[data[10] == 0]
		data_mal = data.loc[data[10] >= 1]
		
		# unpack dataframe
		ben_x = data_ben.iloc[:, :10].to_numpy()
		ben_y = data_ben.iloc[:, 10].to_numpy()
		ben_mask = data_ben.iloc[:, 11:].to_numpy()
		mal_x = data_mal.iloc[:, :10].to_numpy()
		mal_y = data_mal.iloc[:, 10].to_numpy()
		mal_mask = data_mal.iloc[:, 11:].to_numpy()
		mal_yt = np.zeros(len(mal_y))
		
		# generate samples
		ben_adv_x = self.pgd_untargeted.generate(ben_x, mask=ben_mask)
		mal_adv_x = self.pgd_targeted.generate(mal_x, mal_yt, mask=mal_mask)
		
		return np.concatenate([ben_adv_x, mal_adv_x])


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
BATCH_SIZE = 32

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
			attack,
			model,
			criterion,
			optimizer,
			FEATURES_DIM,
			LABELS_DIM,
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
	with open('bids_advpgd_scheduled_history.pkl', 'wb') as f:
		pickle.dump(history, f)
