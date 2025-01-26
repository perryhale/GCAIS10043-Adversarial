
"""

Experimental and unfinished, currently does not 
run because TensorFlowV2Classifier does not 
support binary classifier configuration

"""


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

# type: (np.ndarray, np.ndarray) -> tuple[tuple[np.ndarray]]
def prepare_mask(data_x, data_y):
	
	# merge convert and ben/mal split
	data = pd.DataFrame(np.concatenate([data_x, data_y.reshape((data_y.shape[0], 1,))], axis=-1))
	data_ben = data.loc[data[10] == 0]
	data_mal = data.loc[data[10] >= 1]
	
	# supervised split
	ben_x = data_ben.iloc[:, :-1].to_numpy()
	ben_y = data_ben.iloc[:, -1].to_numpy()
	mal_x = data_mal.iloc[:, :-1].to_numpy()
	mal_y = data_mal.iloc[:, -1].to_numpy()
	mal_yt = np.zeros(len(mal_y))
	
	# generate masks
	ben_mask = np.apply_along_axis(mask_fn, axis=1, arr=ben_x)
	mal_mask = np.apply_along_axis(mask_fn, axis=1, arr=mal_x)
	
	return (ben_x, ben_y, ben_mask), (mal_x, mal_y, mal_yt, mal_mask)

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
	data = data.sample(frac=1, random_state=key)[:8192]#[:10_000_000] ###! truncation for hardware
	
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

# type: (int, int, int, int, str, float, str) -> tf.keras.Model
def get_binary_mlp(
		key, 
		input_dim, 
		hidden_dim, 
		hidden_depth, 
		hidden_act='relu', 
		l2_lambda=0.0, 
		name='Binary-MLP'
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
		1, 
		activation='sigmoid', 
		kernel_initializer=initializers.GlorotUniform(seed=keys[-1]), 
		kernel_regularizer=regularizers.l2(l2_lambda), 
		name='output'
	)(model_y)
	
	model = tf.keras.Model(model_x, model_y, name=name)
	return model

# type: () -> 
###! no straightforward way to make ART deterministic
###! currently, should set global seed to control randomness
def uniform_adv_train(
		attack,
		model,
		criterion,
		optimizer,
		input_dim,
		output_dim,
		feature_res,
		train_sets,
		val_sets,
		epochs=10,
		batch_size=64,
		callbacks=None,
		verbose=False
	):
	
	# assertions
	assert model.optimizer is not None and model.loss is not None and model.metrics is not None, "model must be precompiled"
	
	# unpack
	train_x, train_y, train_mask = train_sets
	val_x, val_y, val_mask = val_sets
	
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
	with tqdm(range(epochs), desc='Train', unit='epoch') as bar:
		for i in bar:
			
			# generate adversarial samples
			train_adv_x = enforce_res(attack.generate(train_x, mask=train_mask), feature_res) ###! ND seed
			val_adv_x = enforce_res(attack.generate(val_x, mask=val_mask), feature_res) ###! ND seed
			if verbose:
				print(f'[Elapsed time: {time.time()-T0:.2f}s]')
			
			# uniform-randomly scale isolated perturbations
			train_xres = (train_adv_x - train_x) * np.random.uniform(0, 1, (train_x.shape[0], 1)) ###! ND seed
			val_xres = (val_adv_x - val_x) * np.random.uniform(0, 1, (val_x.shape[0], 1)) ###! ND seed
			if verbose:
				print(f'[Elapsed time: {time.time()-T0:.2f}s]')
			
			# add perturbations to natural samples
			train_adv_x = train_x + train_xres
			val_adv_x = val_x + val_xres
			if verbose:
				print(f'[Elapsed time: {time.time()-T0:.2f}s]')
			
			# concatenate all samples
			train_x_aug = np.concatenate([train_x, train_adv_x])
			train_y_aug = np.concatenate([train_y, train_y])
			if verbose:
				print(f'[Elapsed time: {time.time()-T0:.2f}s]')
			
			# fit to extended set
			epoch_hist = model.fit(train_x_aug, train_y_aug, epochs=1, batch_size=batch_size, validation_data=(val_x, val_y), callbacks=callbacks, verbose=int(verbose)) ###! ND seed
			if verbose:
				print(f'[Elapsed time: {time.time()-T0:.2f}s]')
			
			# evaluate adversarial accuracy
			train_adv_yh = model.predict(train_adv_x, batch_size=BATCH_SIZE, verbose=int(verbose))
			val_adv_yh = model.predict(val_adv_x, batch_size=BATCH_SIZE, verbose=int(verbose))
			train_adv_accuracy = accuracy_score(train_y, np.argmax(train_adv_yh, axis=-1))
			val_adv_accuracy = accuracy_score(val_y, np.argmax(val_adv_yh, axis=-1))
			
			# record results
			adv_history = dict(
				adv_accuracy=[train_adv_accuracy],
				val_adv_accuracy=[val_adv_accuracy]
			)
			for k in epoch_hist.history.keys():
				train_history[k].extend(epoch_hist.history[k])
			for k in adv_history.keys():
				train_history[k].extend(adv_history[k])
			if verbose:
				print(f'Epoch {i+1}: '+', '.join([f'{k}={train_history[k][-1]}' for k in train_history.keys()]))
				print(f'[Elapsed time: {time.time()-T0:.2f}s]')
			
			bar.set_postfix(
				loss=f'{train_history["loss"][-1]:.4f}', 
				accuracy=f'{train_history["accuracy"][-1]:.4f}', 
				val_loss=f'{train_history["val_loss"][-1]:.4f}', 
				val_accuracy=f'{train_history["val_accuracy"][-1]:.4f}'
			)
	
	return train_history


### hyperparams

# data
FEATURES_RES = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255]).astype('float32')
BATCH_SIZE = 16

# architecture
FEATURES_DIM = 10
LABELS_DIM = 1
HIDDEN_DIM = 16
HIDDEN_DEPTH = 4
HIDDEN_ACT = 'relu' # must be str for name formatting

# training
LEARNING_RATE = 0.001
L2_LAMBDA = 0.001
NUM_EPOCHS = 5
PGD_MAX = 0.1
PGD_ITER = 7
VERBOSE = False


### prepare data

# load and partition data
(train_x, train_y), (val_x, val_y), (test_x, test_y) = get_car_hacking_dataset(K1, binary=True)

# generate masks
train_mask = np.apply_along_axis(mask_fn, axis=1, arr=train_x)
val_mask = np.apply_along_axis(mask_fn, axis=1, arr=val_x)

# normalise data
train_x = train_x / FEATURES_RES
val_x = val_x / FEATURES_RES
test_x = test_x / FEATURES_RES

# undersample benign class in train set
train_rus = RandomUnderSampler(random_state=K1)
train_x, train_y = train_rus.fit_resample(train_x, train_y)
train_mask = train_mask[train_rus.sample_indices_]

# repack
train_sets = (train_x, train_y, train_mask)
val_sets = (val_x, val_y, val_mask)

# trace
print(train_x.shape, train_y.shape, 'train')
print(val_x.shape, val_y.shape, 'val')
print(test_x.shape, test_y.shape, 'test')
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### initialise model

# init and compile
model = get_binary_mlp(
	K2, 
	FEATURES_DIM, 
	HIDDEN_DIM, 
	HIDDEN_DEPTH, 
	hidden_act=HIDDEN_ACT, 
	l2_lambda=L2_LAMBDA, 
	name=f'BIDS_Binary_{HIDDEN_DIM}x{HIDDEN_DEPTH}_{HIDDEN_ACT}_UPGDT{PGD_MAX:.2f}'.replace('.','_')
)
criterion = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE)
model.compile(loss=criterion, optimizer=optimizer, metrics=['binary_accuracy'])

# trace
model.summary()
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### train model

####! set global RNG seed
np.random.seed(K3)
tf.random.set_seed(split_key(K3)[1])

# init art wrapper
art_model = TensorFlowV2Classifier(
	model=model, 
	input_shape=(FEATURES_DIM,), 
	nb_classes=LABELS_DIM, 
	loss_object=criterion, 
	optimizer=optimizer, 
	clip_values=(0,1)
)
pgd_untargeted = ProjectedGradientDescent(
	estimator=art_model, 
	eps=PGD_MAX, 
	eps_step=PGD_MAX/PGD_ITER, 
	max_iter=PGD_ITER, 
	num_random_init=1, 
	targeted=False, 
	batch_size=BATCH_SIZE, 
	verbose=VERBOSE
)

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
	train_history = uniform_adv_train(
		pgd_untargeted,
		model,
		criterion,
		optimizer,
		FEATURES_DIM,
		LABELS_DIM,
		FEATURES_RES,
		(train_x, train_y, train_mask),
		(val_x, val_y, val_mask),
		epochs=NUM_EPOCHS,
		batch_size=BATCH_SIZE,
		pgd_max=PGD_MAX,
		callbacks=[checkpoint_callback],
		verbose=VERBOSE
	)

# trace
finally:
	print(f'[Elapsed time: {time.time()-T0:.2f}s]')
	print(train_history)


### evaluate model

test_yh = model.predict(test_x, batch_size=BATCH_SIZE)
test_loss = criterion(test_y, test_yh).numpy()
test_accuracy = accuracy_score(test_y, np.argmax(test_yh, axis=-1))
test_cfm = confusion_matrix(test_y, np.argmax(test_yh, axis=-1), labels=range(LABELS_DIM))
test_history = dict(loss=test_loss, accuracy=test_accuracy, cfm=test_cfm)
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')
print(test_cfm)


### save results

# dump history
with open('bids_binary_advpgd_uniform_history.pkl', 'wb') as f:
	pickle.dump(dict(train=train_history, test=test_history), f)

# init figure
fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
ax1, ax2, = axis

# plot loss curves
ax1.plot(train_history['loss'], label='train')
ax1.plot(train_history['val_loss'], label='validation', c='r')
ax1.scatter([len(train_history['loss'])-1], [test_loss], marker='x', label='test', c='g')
ax1.set_ylabel('Loss')

# plot accuracy curves
ax2.plot(train_history['accuracy'], label='train', c='blue')
ax2.plot(train_history['val_accuracy'], label='validation', c='r')
ax2.plot(train_history['adv_accuracy'], label='train adversarial', c='blue', linestyle='dashed')
ax2.plot(train_history['val_adv_accuracy'], label='val adversarial', c='r', linestyle='dashed')
ax2.scatter([len(train_history['accuracy'])-1], [test_accuracy], marker='x', label='test', c='g')
ax2.set_ylabel('Accuracy')

# set common features
for ax in axis:
	ax.set_xlabel('Epoch')
	ax.grid()
	ax.legend()

# adjust and save figure
plt.show()
#plt.subplots_adjust(left=0.04, right=0.96, bottom=0.18, wspace=0.3)
#plt.savefig('bids_advpgd_uniform_train.png')
