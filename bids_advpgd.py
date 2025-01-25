import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix, accuracy_score

# attack
import art
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import ProjectedGradientDescent

# util
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
#import matplotlib.pyplot as plt


### set RNG seed

RNG_SEED = 0
tf.keras.utils.set_random_seed(RNG_SEED)
np.random.seed(RNG_SEED)


### functions

# type: (np.ndarray) -> np.ndarray
def mask_fn(x):
	mask = np.zeros(x.shape)
	mask[2:2+x[1]] = 1.
	return mask

# type: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
def enforce_res(xs, res, mask=None):
	res = xs - np.round(np.minimum(res, np.maximum(0., xs * res))) / res
	if mask is not None:
		res *= mask
	return xs - res

# type: (pd.DataFrame, float, float) -> tuple[pd.DataFrame]
def train_val_test_split(data, train_ratio, val_ratio):
	train_data = data.iloc[:int(train_ratio*len(data.index)), :]
	val_data = data.iloc[int(train_ratio*len(data.index)):int((train_ratio+val_ratio)*len(data.index)), :]
	test_data = data.iloc[int((train_ratio+val_ratio)*len(data.index)):, :]
	return (train_data, val_data, test_data)

# type: (pd.DataFrame) -> tuple[np.ndarray]
def standard_split(data, norm_res):
	x = data.iloc[:, :-1].to_numpy() / norm_res
	y = data.iloc[:, -1].to_numpy()
	return (x, y)

# type: (pd.DataFrame) -> tuple[tuple[np.ndarray]]
def targeted_split(data, norm_res):
	data_ben = data.loc[data[10] == 0]
	data_mal = data.loc[data[10] >= 1]
	ben_x = data_ben.iloc[:, :-1].to_numpy() / norm_res
	ben_y = data_ben.iloc[:, -1].to_numpy()
	mal_x = data_mal.iloc[:, :-1].to_numpy() / norm_res
	mal_y = data_mal.iloc[:, -1].to_numpy()
	mal_yt = np.zeros(len(mal_y))
	ben_mask = np.apply_along_axis(mask_fn, axis=1, arr=data_ben.iloc[:, :-1].to_numpy())
	mal_mask = np.apply_along_axis(mask_fn, axis=1, arr=data_mal.iloc[:, :-1].to_numpy())
	return ((ben_x, ben_y, ben_mask), (mal_x, mal_y, mal_yt, mal_mask))

# type: (str, dict[str:List], str, str) -> None
def write_dict_csv(path_name, dictionary, header=None, mode='w'):
	try:
		with open(path_name, mode) as f:
			if header is not None: 
				f.write(header+'\n')
			f.write(', '.join(dictionary.keys())+'\n')
			for epoch in zip(*[dictionary[k] for k in dictionary.keys()]):
				f.write(', '.join([str(v) for v in epoch])+'\n')
		return True
	except Exception as error:
		print(error)
		return False


### hyperparameters

# data partitioning
TRAIN_RATIO = 0.7
VAL_RATIO = 0.05
TEST_RATIO = 1 - (TRAIN_RATIO + VAL_RATIO)
BATCH_SIZE = 512
BATCH_SIZE_PGD = 2048

# data dimensions
D_FEATURES = 10
D_LABELS = 5
RES_FEATURES = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255])

# training
LR = 0.001
L2_LAM = 0.001

OUTER_EPOCH = 2
PGD_EPOCH = 8
INNER_EPOCH = 2

PGD_ADV = 1
PGD_ITER = 7
PGD_EPS_MIN = 1e-9
PGD_EPS_MAX = 0.5

VERBOSE = False

# model
NAME = f'bids_advpgd{OUTER_EPOCH}{INNER_EPOCH}{PGD_EPOCH}'#_os'
HIDDEN_ACT = 'relu'


### initialise dataset

# load data and shuffle
data = pd.read_csv('car_hacking_dataset/car_hacking_dataset.csv', header=None)
data = data.sample(frac=1)

# partition dataset
train_data, val_data, test_data = train_val_test_split(data[:10_000_000], TRAIN_RATIO, VAL_RATIO)
val_x, val_y = standard_split(val_data, RES_FEATURES)
test_x, test_y = standard_split(test_data, RES_FEATURES)

# oversample train set
#train_x, train_y = standard_split(train_data, RES_FEATURES)
#train_x, train_y = RandomOverSampler().fit_resample(train_x, train_y)
#train_data = pd.DataFrame(np.concatenate([(train_x * RES_FEATURES).astype(np.int32), train_y.reshape((train_y.shape[0],1))], axis=-1))

# split train set
(train_ben_x, train_ben_y, train_ben_mask), (train_mal_x, train_mal_y, train_mal_yt, train_mal_mask) = targeted_split(train_data, RES_FEATURES)

# trace
print(train_ben_x.shape, train_ben_y.shape, 'train_ben')
print(train_mal_x.shape, train_mal_y.shape, 'train_mal')
print(val_x.shape, val_y.shape, 'val')
print(test_x.shape, test_y.shape, 'test')

# # convert to tf Dataset
# train_ben_dataset = tf.data.Dataset.from_tensor_slices((train_ben_x, train_ben_y))
# train_mal_dataset = tf.data.Dataset.from_tensor_slices((train_mal_x, train_mal_y))
# train_mal_target_dataset = tf.data.Dataset.from_tensor_slices((train_mal_x, train_mal_yt))

# train_ben_dataset.batch(BATCH_SIZE)
# train_mal_dataset.batch(BATCH_SIZE)
# train_mal_target_dataset.batch(BATCH_SIZE)

# train_ben_dataset.cache().shuffle(2048).prefetch(16)
# train_mal_dataset.cache().shuffle(2048).prefetch(16)
# train_mal_target_dataset.cache().shuffle(2048).prefetch(16)


### initialise model

model_x = layers.Input(shape=(10,), name='input')
model_y = layers.Dense(16, activation=HIDDEN_ACT, kernel_regularizer=regularizers.l2(L2_LAM), name='hidden1')(model_x)
model_y = layers.Dense(16, activation=HIDDEN_ACT, kernel_regularizer=regularizers.l2(L2_LAM), name='hidden2')(model_y)
model_y = layers.Dense(16, activation=HIDDEN_ACT, kernel_regularizer=regularizers.l2(L2_LAM), name='hidden3')(model_y)
model_y = layers.Dense(16, activation=HIDDEN_ACT, kernel_regularizer=regularizers.l2(L2_LAM), name='hidden4')(model_y)
model_y = layers.Dense(5, activation='softmax', kernel_regularizer=regularizers.l2(L2_LAM), name='output')(model_y)
model = tf.keras.Model(model_x, model_y, name=NAME)
model.summary()

criterion = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.AdamW(learning_rate=LR)
model.compile(loss=criterion, optimizer=optimizer, metrics=['accuracy'])


### train model

# init history
train_history = {
	'loss':[], 
	'accuracy':[], 
	'val_loss':[], 
	'val_accuracy':[]
}

# optimize against data augmented by increasingly strong PGD samples over #PGD_EPOCH increments in an OUTER_EPOCH x INNER_EPOCH schedule
outer_axis = range(OUTER_EPOCH)
pgd_axis = np.linspace(PGD_EPS_MIN, PGD_EPS_MAX, num=PGD_EPOCH)
inner_axis = range(INNER_EPOCH)
with tqdm(outer_axis, desc='ADVT 0 (outer)', unit='epoch') as bar:
	for i in bar:
		for e in tqdm(pgd_axis, desc='ADVT 1 (pgd)  ', unit='epsilon'):
			for j in tqdm(inner_axis, desc='ADVT 2 (inner)', unit='epoch'):
				
				# setup art wrapper
				art_model = TensorFlowV2Classifier(model=model, input_shape=(D_FEATURES,), nb_classes=D_LABELS, loss_object=criterion, optimizer=optimizer, clip_values=(0,1))
				pgd_untargeted = ProjectedGradientDescent(estimator=art_model, eps=e, eps_step=(e/PGD_ITER), max_iter=PGD_ITER, num_random_init=1, targeted=False, batch_size=BATCH_SIZE_PGD, verbose=VERBOSE)
				pgd_targeted = ProjectedGradientDescent(estimator=art_model, eps=e, eps_step=(e/PGD_ITER), max_iter=PGD_ITER, num_random_init=1, targeted=True, batch_size=BATCH_SIZE_PGD, verbose=VERBOSE)
				
				# generate adversarial samples
				###! verify listcomp values are distinct ie [rand forall i]
				train_ben_adv_x = enforce_res(np.concatenate([pgd_untargeted.generate(train_ben_x, mask=train_ben_mask) for i in range(PGD_ADV)]), RES_FEATURES) 
				train_mal_adv_x = enforce_res(np.concatenate([pgd_targeted.generate(train_mal_x, train_mal_yt, mask=train_mal_mask) for i in range(PGD_ADV)]), RES_FEATURES)
				
				# concatenate all samples
				epoch_x = np.concatenate([train_ben_x, train_mal_x, train_ben_adv_x, train_mal_adv_x])
				epoch_y = np.concatenate([train_ben_y, train_mal_y, np.concatenate([train_ben_y for i in range(PGD_ADV)]), np.concatenate([train_mal_y for i in range(PGD_ADV)])])
				
				# # convert to tf Dataset
				# dataset = tf.data.Dataset.from_tensor_slices((epoch_x, epoch_y))
				# dataset.batch(BATCH_SIZE)
				# dataset.cache().shuffle(2048).prefetch(16)
				
				# fit to extended set
				epoch_hist = model.fit(epoch_x, epoch_y, epochs=1, batch_size=BATCH_SIZE, validation_data=(val_x, val_y), verbose=int(VERBOSE))
				
				# update history
				for k in train_history.keys():
					train_history[k].extend(epoch_hist.history[k])
		
		bar.set_postfix(
			loss=f'{train_history["loss"][-1]:.4f}',
			val_loss=f'{train_history["val_loss"][-1]:.4f}'
		)


### evaluate model

test_yh = model.predict(test_x, batch_size=BATCH_SIZE)
test_loss = criterion(test_y, test_yh).numpy()
test_accuracy = accuracy_score(test_y, np.argmax(test_yh, axis=-1))
test_cfm = confusion_matrix(test_y, np.argmax(test_yh, axis=-1), labels=range(5))
test_history = {'loss':[test_loss], 'accuracy':[test_accuracy], 'cfm':[test_cfm]}
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')
print(test_cfm)


### save results

model.save_weights(NAME+'.weights.h5')
write_dict_csv(f'{NAME}_train_history.csv', train_history)
write_dict_csv(f'{NAME}_test_history.csv', test_history)

# # plot training history
# plt.plot(history['loss'], label='train')
# plt.plot(history['val_loss'], label='validation', c='r')
# plt.scatter([len(history['loss'])-1], [test_loss], marker='x', label='test', c='g')
# plt.legend()
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# #plt.show()
# plt.savefig(NAME+'.train.png')
