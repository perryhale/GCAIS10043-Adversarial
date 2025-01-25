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
#from imblearn.over_sampling import RandomOverSampler



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

# type: (np.ndarray, np.ndarray) -> tuple[tuple[np.ndarray]]
def benmal_split(data_x, data_y):
	
	# type: (np.ndarray) -> np.ndarray
	def mask_fn(x):
		mask = np.zeros(x.shape)
		mask[2:2+x[1]] = 1.
		return mask
	
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
	data = data.sample(frac=1, random_state=key)[:10_000_000] ###! truncation for hardware
	
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
###! aug_ratio is unimplemented because of concerns about non-random sampling with slicing
def uniform_pgd_benmal_train(
		model, 
		criterion, 
		optimizer, 
		input_dim, 
		output_dim, 
		feature_res, 
		train_ben_sets,
		train_mal_sets,
		val_ben_sets,
		val_mal_sets,
		epochs=10, 
		batch_size=64, 
		pgd_iter=7, 
		pgd_eps=0.1, 
		#aug_ratio=1.0,
		callbacks=None,
		verbose=False
	):
	
	# assertions
	assert model.optimizer is not None and model.loss is not None and model.metrics is not None, "model must be precompiled"
	assert pgd_iter > 0, "pgd_iter must be in Z+"
	assert pgd_eps <= 1.0 and pgd_eps > 0, "pgd_eps must be in (0..1]"
	#assert aug_ratio >= 0.0 and aug_ratio <= 1.0, "aug_ratio must be in [0..1]"
	
	# unpack
	train_ben_x, train_ben_y, train_ben_mask = train_ben_sets
	train_mal_x, train_mal_y, train_mal_yt, train_mal_mask = train_mal_sets
	val_ben_x, val_ben_y, val_ben_mask = val_ben_sets
	val_mal_x, val_mal_y, val_mal_yt, val_mal_mask = val_mal_sets
	
	# init history
	train_history = {
		
		# train set
		'loss':[],
		'accuracy':[],
		'ben_adv_accuracy':[],
		'mal_adv_accuracy':[],
		
		# val set
		'val_loss':[],
		'val_accuracy':[],
		'val_ben_adv_accuracy':[],
		'val_mal_adv_accuracy':[]
	}
	
	# init art wrapper
	art_model = TensorFlowV2Classifier(
		model=model, 
		input_shape=(input_dim,), 
		nb_classes=output_dim, 
		loss_object=criterion, 
		optimizer=optimizer, 
		clip_values=(0,1)
	)
	pgd_untargeted = ProjectedGradientDescent(
		estimator=art_model, 
		eps=pgd_eps, 
		eps_step=pgd_eps/pgd_iter, 
		max_iter=pgd_iter, 
		num_random_init=1, 
		targeted=False, 
		batch_size=batch_size, 
		verbose=verbose
	)
	pgd_targeted = ProjectedGradientDescent(
		estimator=art_model, 
		eps=pgd_eps, 
		eps_step=pgd_eps/pgd_iter, 
		max_iter=pgd_iter, 
		num_random_init=1, 
		targeted=True, 
		batch_size=batch_size, 
		verbose=verbose
	)
	
	# train model
	with tqdm(range(epochs), desc='Train', unit='epoch') as bar:
		for i in bar:
			
			# generate adversarial samples
			train_ben_x_adv = enforce_res(pgd_untargeted.generate(train_ben_x, mask=train_ben_mask), feature_res) ###! ND seed
			train_mal_x_adv = enforce_res(pgd_targeted.generate(train_mal_x, train_mal_yt, mask=train_mal_mask), feature_res) ###! ND seed
			val_ben_x_adv = enforce_res(pgd_untargeted.generate(val_ben_x, mask=val_ben_mask), feature_res) ###! ND seed
			val_mal_x_adv = enforce_res(pgd_targeted.generate(val_mal_x, val_mal_yt, mask=val_mal_mask), feature_res) ###! ND seed
			if verbose:
				print(f'[Elapsed time: {time.time()-T0:.2f}s]')
			
			# uniform-randomly scale isolated perturbations
			train_ben_xres = (train_ben_x_adv - train_ben_x) * np.random.uniform(0, 1, (train_ben_x.shape[0], 1)) ###! ND seed
			train_mal_xres = (train_mal_x_adv - train_mal_x) * np.random.uniform(0, 1, (train_mal_x.shape[0], 1)) ###! ND seed
			val_ben_xres = (val_ben_x_adv - val_ben_x) * np.random.uniform(0, 1, (val_ben_x.shape[0], 1)) ###! ND seed
			val_mal_xres = (val_mal_x_adv - val_mal_x) * np.random.uniform(0, 1, (val_mal_x.shape[0], 1)) ###! ND seed
			if verbose:
				print(f'[Elapsed time: {time.time()-T0:.2f}s]')
			
			# add perturbations to natural samples
			train_ben_x_adv = train_ben_x + train_ben_xres
			train_mal_x_adv = train_mal_x + train_mal_xres
			val_ben_x_adv = val_ben_x + val_ben_xres
			val_mal_x_adv = val_mal_x + val_mal_xres
			if verbose:
				print(f'[Elapsed time: {time.time()-T0:.2f}s]')
			
			# concatenate all samples
			train_x = np.concatenate([train_ben_x, train_mal_x, train_ben_x_adv, train_mal_x_adv])
			train_y = np.concatenate([train_ben_y, train_mal_y, train_ben_y, train_mal_y])
			val_nat_x = np.concatenate([val_ben_x, val_mal_x])
			val_nat_y = np.concatenate([val_ben_y, val_mal_y])
			if verbose:
				print(f'[Elapsed time: {time.time()-T0:.2f}s]')
			
			# fit to extended set
			epoch_hist = model.fit(train_x, train_y, epochs=1, batch_size=batch_size, validation_data=(val_nat_x, val_nat_y), callbacks=callbacks, verbose=int(verbose)) ###! ND seed
			if verbose:
				print(f'[Elapsed time: {time.time()-T0:.2f}s]')
			
			# evaluate adversarial accuracy
			train_ben_yh_adv = model.predict(train_ben_x_adv, batch_size=BATCH_SIZE, verbose=int(verbose))
			train_mal_yh_adv = model.predict(train_mal_x_adv, batch_size=BATCH_SIZE, verbose=int(verbose))
			val_ben_yh_adv = model.predict(val_ben_x_adv, batch_size=BATCH_SIZE, verbose=int(verbose))
			val_mal_yh_adv = model.predict(val_mal_x_adv, batch_size=BATCH_SIZE, verbose=int(verbose))
			train_ben_adv_accuracy = accuracy_score(train_ben_y, np.argmax(train_ben_yh_adv, axis=-1))
			train_mal_adv_accuracy = accuracy_score(train_mal_y, np.argmax(train_mal_yh_adv, axis=-1))
			val_ben_adv_accuracy = accuracy_score(val_ben_y, np.argmax(val_ben_yh_adv, axis=-1))
			val_mal_adv_accuracy = accuracy_score(val_mal_y, np.argmax(val_mal_yh_adv, axis=-1))
			
			# record results
			adv_history = dict(
				ben_adv_accuracy=[train_ben_adv_accuracy],
				mal_adv_accuracy=[train_mal_adv_accuracy],
				val_ben_adv_accuracy=[val_ben_adv_accuracy],
				val_mal_adv_accuracy=[val_mal_adv_accuracy]
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
BATCH_SIZE = 512

# architecture
FEATURES_DIM = 10
LABELS_DIM = 5
HIDDEN_DIM = 16
HIDDEN_DEPTH = 4
HIDDEN_ACT = 'relu' # must be str for name formatting

# training
LEARNING_RATE = 0.001
L2_LAMBDA = 0.001
NUM_EPOCHS = 5
PGD_EPS = 0.1


### prepare data

# load and partition data
train_sets, val_sets, (test_x, test_y) = get_car_hacking_dataset(K1)
(train_ben_x, train_ben_y, train_ben_mask), (train_mal_x, train_mal_y, train_mal_yt, train_mal_mask) = benmal_split(*train_sets)
(val_ben_x, val_ben_y, val_ben_mask), (val_mal_x, val_mal_y, val_mal_yt, val_mal_mask) = benmal_split(*val_sets)

# normalise data
train_ben_x = train_ben_x / FEATURES_RES
train_mal_x = train_mal_x / FEATURES_RES
val_ben_x = val_ben_x / FEATURES_RES
val_mal_x = val_mal_x / FEATURES_RES
test_x = test_x / FEATURES_RES

# undersample benign class in train set
train_undersample_indices = np.random.default_rng(seed=K1).permutation(len(train_ben_x))[:len(train_mal_x)//4]
train_ben_x = train_ben_x[train_undersample_indices]
train_ben_y = train_ben_y[train_undersample_indices]
train_ben_mask = train_ben_mask[train_undersample_indices]

# repack
train_ben_sets = (train_ben_x, train_ben_y, train_ben_mask)
train_mal_sets = (train_mal_x, train_mal_y, train_mal_yt, train_mal_mask)
val_ben_sets = (val_ben_x, val_ben_y, val_ben_mask)
val_mal_sets = (val_mal_x, val_mal_y, val_mal_yt, val_mal_mask)


# trace
print(train_ben_x.shape, train_ben_y.shape, 'train_ben')
print(train_mal_x.shape, train_mal_y.shape, 'train_mal')
print(val_ben_x.shape, val_ben_y.shape, 'val_ben')
print(val_mal_x.shape, val_mal_y.shape, 'val_mal')
print(test_x.shape, test_y.shape, 'test')
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### initialise model

# init and compile
model = get_multiclass_mlp(
	K2, 
	FEATURES_DIM, 
	LABELS_DIM, 
	HIDDEN_DIM, 
	HIDDEN_DEPTH, 
	hidden_act=HIDDEN_ACT, 
	l2_lambda=L2_LAMBDA, 
	name=f'BIDS_{HIDDEN_DIM}x{HIDDEN_DEPTH}_{HIDDEN_ACT}_UPGDT{PGD_EPS:.2f}'.replace('.','_')
)
criterion = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE)
model.compile(loss=criterion, optimizer=optimizer, metrics=['accuracy'])

# trace
model.summary()
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### train model

####! set global RNG seed
np.random.seed(K3)
tf.random.set_seed(split_key(K3)[1])

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
	train_history = uniform_pgd_benmal_train(
		model,
		criterion,
		optimizer,
		FEATURES_DIM,
		LABELS_DIM,
		FEATURES_RES,
		train_ben_sets,
		train_mal_sets,
		val_ben_sets,
		val_mal_sets,
		epochs=NUM_EPOCHS,
		batch_size=BATCH_SIZE,
		pgd_eps=PGD_EPS,
		callbacks=[checkpoint_callback]
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

"""
	train_history = {
		
		# train set
		'loss':[],
		'accuracy':[],
		'ben_adv_accuracy':[],
		'mal_adv_accuracy':[],
		
		# val set
		'val_loss':[],
		'val_accuracy':[],
		'val_ben_adv_accuracy':[],
		'val_mal_adv_accuracy':[]
	}
"""

# dump history
with open('bids_advpgd_uniform_history.pkl', 'wb') as f:
	pickle.dump(dict(train=train_history, test=test_history), f)

# init figure
fig, axis = plt.subplots(nrows=1, ncols=3, figsize=(14,6))
ax1, ax2, ax3 = axis

# plot loss curves
ax1.plot(train_history['loss'], label='train')
ax1.plot(train_history['val_loss'], label='validation', c='r')
ax1.scatter([len(train_history['loss'])-1], [test_loss], marker='x', label='test', c='g')

# plot accuracy curves
ax2.plot(train_history['accuracy'], label='train')
ax2.plot(train_history['val_accuracy'], label='validation', c='r')
ax2.scatter([len(train_history['accuracy'])-1], [test_accuracy], marker='x', label='test', c='g')

# plot adversarial accuracy
ax3.plot(train_history['ben_adv_accuracy'], label='train_ben_adv_accuracy', c='brown', linestyle='dashed')
ax3.plot(train_history['mal_adv_accuracy'], label='train_mal_adv_accuracy', c='purple', linestyle='dashed')
ax3.plot(train_history['val_ben_adv_accuracy'], label='val_ben_adv_accuracy', c='brown')
ax3.plot(train_history['val_mal_adv_accuracy'], label='val_mal_adv_accuracy', c='purple')

# set labels
ax1.set_ylabel('Loss')
ax2.set_ylabel('Baseline accuracy')
ax3.set_ylabel('Adversarial accuracy')

# set legends
ax1.legend()
ax3.legend()

# set common features
for ax in axis[1:]:
	ax.set_xlabel('Epoch')
	ax.set_ylim(0,1)
	ax.grid()

# adjust and save figure
plt.subplots_adjust(left=0.04, right=0.96, bottom=0.18, wspace=0.3)
plt.show()
#plt.savefig('bids_advpgd_uniform_train.png')
