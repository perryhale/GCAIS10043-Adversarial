import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

import art
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import ProjectedGradientDescent

from sklearn.metrics import confusion_matrix, accuracy_score


### set RNG seed
RNG_SEED = 0
tf.keras.utils.set_random_seed(RNG_SEED)
np.random.seed(RNG_SEED)


### data setup
TRAIN_RATIO = 0.7
VAL_RATIO = 0.05
TEST_RATIO = 1 - TRAIN_RATIO + VAL_RATIO

FEATURE_SCALE = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255])

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
def train_val_test_split(data):
	train_data = data.iloc[:int(TRAIN_RATIO*len(data.index)), :]
	val_data = data.iloc[int(TRAIN_RATIO*len(data.index)):int((TRAIN_RATIO+VAL_RATIO)*len(data.index)), :]
	test_data = data.iloc[int((TRAIN_RATIO+VAL_RATIO)*len(data.index)):, :]
	return (train_data, val_data, test_data)

# type: (pd.DataFrame) -> tuple[np.ndarray]
def standard_split(data):
	x = data.iloc[:, :-1].to_numpy() / FEATURE_SCALE
	y = data.iloc[:, -1].to_numpy()
	return (x, y)

# type: (pd.DataFrame) -> tuple[tuple[np.ndarray]]
def adversarial_split(data):
	data_ben = data.loc[data[10] == 0]
	data_mal = data.loc[data[10] >= 1]
	ben_x = data_ben.iloc[:, :-1].to_numpy() / FEATURE_SCALE
	ben_y = data_ben.iloc[:, -1].to_numpy()
	mal_x = data_mal.iloc[:, :-1].to_numpy() / FEATURE_SCALE
	mal_y = data_mal.iloc[:, -1].to_numpy()
	mal_yt = np.zeros(len(mal_y))
	ben_mask = np.apply_along_axis(mask_fn, axis=1, arr=data_ben.iloc[:, :-1].to_numpy())
	mal_mask = np.apply_along_axis(mask_fn, axis=1, arr=data_mal.iloc[:, :-1].to_numpy())
	return ((ben_x, ben_y, ben_mask), (mal_x, mal_y, mal_yt, mal_mask))

data = pd.read_csv('car_hacking_dataset/car_hacking_dataset.csv', header=None)
data = data.sample(frac=1)
train_data, val_data, test_data = train_val_test_split(data)
(train_ben_x, train_ben_y, train_ben_mask), (train_mal_x, train_mal_y, train_mal_yt, train_mal_mask) = adversarial_split(train_data)
val_x, val_y = standard_split(val_data)
test_x, test_y = standard_split(test_data)

train_ben_x = train_ben_x[:int(0.25*len(train_mal_x))] # undersampling
train_ben_y = train_ben_y[:int(0.25*len(train_mal_y))]
train_ben_mask = train_ben_mask[:int(0.25*len(train_mal_mask))]

print(train_ben_x.shape, train_ben_y.shape, 'train (benign)')
print(train_mal_x.shape, train_mal_y.shape, 'train (malicious)')
print(val_x.shape, val_y.shape, 'validation')
print(test_x.shape, test_y.shape, 'test')


### define model
#NAME = 'baseline_ids_tfv2_pgd_train' # natural sample, outer 1, res 32, inner 1
#NAME = 'baseline_ids_tfv2_pgd_train_us' # undersample, outer 1, res 32, inner 1
#NAME = 'baseline_ids_tfv2_pgd_train_us_i5.weights.h5' # undersample, outer 1, res 32, inner 5
#NAME = 'baseline_ids_tfv2_pgd_train_us_e5.weights.h5' # undersample, outer 5, res 32, inner 1
NAME = 'baseline_ids_tfv2_pgd_train_us_mx0_25.weights.h5' # undersample, outer 1, res 8, max 0.25, inner 1
HIDDEN_ACT = 'relu'
L2_LAM = 0.001
LR = 0.001

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.AdamW(learning_rate=LR)

model_x = layers.Input(shape=(10,), name='input')
model_y = layers.Dense(16, activation=HIDDEN_ACT, kernel_regularizer=regularizers.l2(L2_LAM), name='hidden1')(model_x)
model_y = layers.Dense(16, activation=HIDDEN_ACT, kernel_regularizer=regularizers.l2(L2_LAM), name='hidden2')(model_y)
model_y = layers.Dense(16, activation=HIDDEN_ACT, kernel_regularizer=regularizers.l2(L2_LAM), name='hidden3')(model_y)
model_y = layers.Dense(16, activation=HIDDEN_ACT, kernel_regularizer=regularizers.l2(L2_LAM), name='hidden4')(model_y)
model_y = layers.Dense(5, activation='softmax', kernel_regularizer=regularizers.l2(L2_LAM), name='output')(model_y)
model = tf.keras.Model(model_x, model_y, name=NAME)
model.summary()
model.compile(loss=loss_object, optimizer=optimizer, metrics=['accuracy'])


### adversarially train model
OUTER_EPOCHS = 1
INNER_EPOCHS = 1
ADV_COUNT = 1
EPS_RES = 8
EPS_MIN = 1e-9
EPS_MAX = 0.25
PGD_ITER = 3
BATCH_SIZE = 512
VERBOSE = True

history = {'loss':[], 'val_loss':[], 'accuracy':[], 'val_accuracy':[]}

for i in tqdm(range(OUTER_EPOCHS)) if OUTER_EPOCHS > 1 else [0]:
	for e in tqdm(np.linspace(EPS_MIN, EPS_MAX, num=EPS_RES)):
		for j in tqdm(range(INNER_EPOCHS)) if INNER_EPOCHS > 1 else [0]:
			
			# setup art wrapper
			art_model = TensorFlowV2Classifier(model=model, input_shape=(10,), nb_classes=5, loss_object=loss_object, optimizer=optimizer, clip_values=(0,1))
			pgd_untargeted = ProjectedGradientDescent(estimator=art_model, eps=e, eps_step=(e/PGD_ITER), max_iter=PGD_ITER, num_random_init=1, targeted=False, batch_size=8192, verbose=VERBOSE)
			pgd_targeted = ProjectedGradientDescent(estimator=art_model, eps=e, eps_step=(e/PGD_ITER), max_iter=PGD_ITER, num_random_init=1, targeted=True, batch_size=8192, verbose=VERBOSE)
			
			# generate adversarial samples
			train_ben_adv_x = enforce_res(np.concatenate([pgd_untargeted.generate(train_ben_x, mask=train_ben_mask) for i in range(ADV_COUNT)]), FEATURE_SCALE)
			train_mal_adv_x = enforce_res(np.concatenate([pgd_targeted.generate(train_mal_x, train_mal_yt, mask=train_mal_mask) for i in range(ADV_COUNT)]), FEATURE_SCALE)
			
			# concatenate all samples
			epoch_x = np.concatenate([train_ben_x, train_mal_x, train_ben_adv_x, train_mal_adv_x])
			epoch_y = np.concatenate([train_ben_y, train_mal_y, np.concatenate([train_ben_y for i in range(ADV_COUNT)]), np.concatenate([train_mal_y for i in range(ADV_COUNT)])])
			
			# fit to extended set
			epoch_hist = model.fit(epoch_x, epoch_y, epochs=1, batch_size=BATCH_SIZE, validation_data=(val_x, val_y), verbose=int(VERBOSE))
			
			# update history
			for k in history.keys():
				history[k].extend(epoch_hist.history[k])


### evaluate model
test_yh = model.predict(test_x)

test_loss = loss_object(test_y, test_yh).numpy()
test_accuracy = accuracy_score(test_y, np.argmax(test_yh, axis=-1))
test_cfm = confusion_matrix(test_y, np.argmax(test_yh, axis=-1), labels=range(5))

print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')
print(test_cfm)
# Test loss: 0.17924007773399353
# Test accuracy: 0.9733608365058899
# [[3777598   17687     121   11715      50]
 # [      0  146934       0       0       0]
 # [  76545    2447   34666    4938    3432]
 # [      0       0       0  149801       0]
 # [      0       0       0       0  163651]]


### save results
plt.plot(history['loss'], label='train')
plt.plot(history['val_loss'], label='validation', c='r')
plt.scatter([len(history['loss'])-1], [test_loss], marker='x', label='test', c='g')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.show()

plt.savefig(NAME+'.png')
model.save_weights(NAME+'.weights.h5')
