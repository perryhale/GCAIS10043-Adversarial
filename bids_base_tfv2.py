import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, accuracy_score


### set RNG seed
RNG_SEED = 0
tf.keras.utils.set_random_seed(RNG_SEED)
np.random.seed(RNG_SEED)


### data preparation
TRAIN_RATIO = 0.7
VAL_RATIO = 0.05
TEST_RATIO = 1 - (TRAIN_RATIO + VAL_RATIO)
FEATURE_SCALE = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255])
BATCH_SIZE = 512

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

# load data and shuffle
data = pd.read_csv('car_hacking_dataset/car_hacking_dataset.csv', header=None)
data = data.sample(frac=1)

# basic partition
train_data, val_data, test_data = train_val_test_split(data)

# train set oversample and standard split
train_x, train_y = standard_split(train_data)
train_x, train_y = RandomOverSampler().fit_resample(train_x, train_y)

# val and test standard split 
val_x, val_y = standard_split(val_data)
test_x, test_y = standard_split(test_data)

print(train_x.shape, train_y.shape, 'train')
print(val_x.shape, val_y.shape, 'validation')
print(test_x.shape, test_y.shape, 'test')

# # convert to tf Dataset
# train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
# val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
# test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

# train_dataset.batch(BATCH_SIZE).cache().shuffle(2048).prefetch(16)
# val_dataset.batch(BATCH_SIZE).cache().shuffle(2048).prefetch(16)
# test_dataset.batch(BATCH_SIZE).cache().shuffle(2048).prefetch(16)


### define model
NAME = 'baseline_ids_tfv2_os'
HIDDEN_ACT = 'relu'
L2_LAM = 0.0
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


### train model
EPOCHS = 3
history = model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(val_x, val_y))


### evaluate model
test_yh = model.predict(test_x, batch_size=BATCH_SIZE)
test_loss = loss_object(test_y, test_yh).numpy()
test_accuracy = accuracy_score(test_y, np.argmax(test_yh, axis=-1))
test_cfm = confusion_matrix(test_y, np.argmax(test_yh, axis=-1), labels=range(5))

print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')
print(test_cfm)


### save results

# save weights
model.save_weights(NAME+'.weights.h5')

# save training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation', c='r')
plt.scatter([len(history.history['loss'])-1], [test_loss], marker='x', label='test', c='g')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.show()
plt.savefig(NAME+'.train.png')

with open(NAME+'.train.csv', 'w') as f:
	f.write(', '.join(history.history.keys()) + '\n')
	for epoch in zip(*[history.history[k] for k in history.history.keys()]):
		f.write(', '.join([str(v) for v in epoch]) + '\n')

# save test results
with open(NAME+'.test.txt', 'w') as f:
	f.write(f'Test loss: {test_loss}\n')
	f.write(f'Test accuracy: {test_accuracy}\n')
	f.write(f'{str(test_cfm)}\n')
