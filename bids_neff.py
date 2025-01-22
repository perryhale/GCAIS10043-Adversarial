import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix, accuracy_score

# util
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle


### functions

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

# type: (np.ndarray, np.ndarray, tf.keras.Model, tf.keras.Loss, float) -> float
def effective_dimension_score(x, y, model, loss_fn, z=1.0, batch_size=32):
	
	# initialise
	x_batches = np.array_split(x, len(x) // batch_size)
	yh = []
	hessian = []
	
	# start gradient tape
	with tf.GradientTape(persistent=True) as tape:
		
		# compute batched predictions
		for batch in x_batches:
			batch_yh = model(batch)
			yh.append(batch_yh)
		yh = tf.concat(yh, axis=0)
		
		# compute first order gradients
		loss = loss_fn(y, yh)
		gradients = tape.gradient(loss, model.trainable_variables)
		gradients = tf.concat([tf.reshape(g, [-1]) for g in gradients], axis=0)
		
		# compute second order gradients
		for g in tqdm(gradients, desc='Hessian computation'):
			hessian_row = tape.gradient(g, model.trainable_variables)
			hessian_row = tf.concat([tf.reshape(h, [-1]) for h in hessian_row], axis=0)
			hessian.append(hessian_row)
	
	# finalise hessian
	hessian = tf.stack(hessian, axis=0)
	
	# compute effective_dimension
	effective_dimension = tf.reduce_sum([e/(e+z) for e in tf.linalg.eigvalsh(hessian)])
	
	return effective_dimension, yh


### hyperparameters

# hardware
BATCH_SIZE = 512

# data partitioning
TRAIN_RATIO = 0.7
VAL_RATIO = 0.05
TEST_RATIO = 1 - (TRAIN_RATIO + VAL_RATIO)

# data dimensions
D_FEATURES = 10
D_LABELS = 5
FEATURES_RES = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255])


### evaluate models

# define names
models = [
	'baseline_ids_tfv2', 
	'baseline_ids_tfv2_os', 
	'baseline_ids_tfv2_pgd_train', 
	'baseline_ids_tfv2_pgd_train_os', 
	'baseline_ids_tfv2_pgd_train_us', 
	'baseline_ids_tfv2_pgd_train_us_e5', 
	'baseline_ids_tfv2_pgd_train_us_i5', 
]

# load dataset
###! convert to tf Dataset
data = pd.read_csv('car_hacking_dataset/car_hacking_dataset.csv', header=None).sample(frac=1)
_, _, test_data = train_val_test_split(data, TRAIN_RATIO, VAL_RATIO)
test_x, test_y = standard_split(test_data, FEATURES_RES)
print(test_x.shape, test_y.shape, 'test')

# run search
history = {}
for model_name in models:
	
	# init model
	model_x = layers.Input(shape=(10,), name='input')
	model_y = layers.Dense(16, activation='relu', name='hidden1')(model_x)
	model_y = layers.Dense(16, activation='relu', name='hidden2')(model_y)
	model_y = layers.Dense(16, activation='relu', name='hidden3')(model_y)
	model_y = layers.Dense(16, activation='relu', name='hidden4')(model_y)
	model_y = layers.Dense(5, activation='softmax', name='output')(model_y)
	model = tf.keras.Model(model_x, model_y, name=model_name)
	model.load_weights(f'models/{model_name}.weights.h5')
	
	# init loss
	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
	
	# compute scores
	test_neff, test_yh = effective_dimension_score(test_x, test_y, model, loss_fn, batch_size=BATCH_SIZE)
	test_loss     = loss_fn(test_y, test_yh).numpy()
	test_accuracy = accuracy_score(test_y, np.argmax(test_yh, axis=-1))
	test_cfm      = confusion_matrix(test_y, np.argmax(test_yh, axis=-1), labels=range(5))
	
	# record results
	history.update({model_name:dict(
		test_neff=test_neff, 
		test_loss=test_loss, 
		test_accuracy=test_accuracy, 
		test_cfm=test_cfm
	)})
	print(', '.join([history[model_name][k] for k in history[model_name].keys()]))

# save history
with open('bids_neff_history.pkl', 'wb') as f:
	pickle.dump(history, f)
