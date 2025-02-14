import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, initializers, regularizers
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .random import split_key


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


# type: (List[tf.keras.Model], List[float], str) -> tf.keras.Model
###! assumes all models are Multiclass-MLPs of the same shape
def merge_multiclass_mlps(models, weights=None, name='Merged Multiclass-MLP'):
	
	# weights setup
	if weights is None:
		weights = np.ones((len(models))) / len(models)
	else:
		msg = f"The number of weights provided must match the number of models. got <{len(models)}> <{len(weights)}>"
		assert len(models)==len(weights), msg
	
	# zero-initialise merged model
	merged_model = tf.keras.models.clone_model(models[0])
	merged_model.name = name
	for layer in merged_model.layers:
		layer.set_weights([np.zeros_like(w) for w in layer.get_weights()])
	
	# accumulate scaled parameters
	for model, weight in zip(models, weights):
		for layer, merge_layer in zip(model.layers, merged_model.layers):
			merge_layer.set_weights([mw+w*weight for w,mw in zip(layer.get_weights(), merge_layer.get_weights())])
	
	return merged_model


# type: (tf.keras.Model, tf.keras.Loss, np.ndarray, np.ndarray, int, bool) -> np.ndarray
def compute_hessian(model, loss_fn, x, y, batch_size=32, verbose=False):
	
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
		for g in tqdm(gradients, desc='Compute hessian') if verbose else gradients:
			hessian_row = tape.gradient(g, model.trainable_variables)
			hessian_row = tf.concat([tf.reshape(h, [-1]) for h in hessian_row], axis=0)
			hessian.append(hessian_row)
	
	# finalise
	hessian = tf.stack(hessian, axis=0).numpy()
	
	return hessian


# type: (tf.keras.Model, tf.keras.losses.Loss, np.ndarray, np.ndarray, int, str, bool) -> Dict[str : {float, np.ndarray}]
def classifier_evaluation(model, loss_fn, x, y, batch_size=32, score_average='weighted', verbose=False):
	
	# predict
	yh = model.predict(x, batch_size=batch_size, verbose=int(verbose))
	yh_index = np.argmax(yh, axis=-1)
	
	# score
	loss = loss_fn(y, yh).numpy()
	accuracy = accuracy_score(y, yh_index)
	precision, recall, fscore, _ = precision_recall_fscore_support(y, yh_index, average=score_average, labels=range(yh.shape[-1]), zero_division=0.0)
	confusion = confusion_matrix(y, yh_index, labels=range(yh.shape[-1]))
	
	# format
	history = {
		'loss':loss.item(),
		'accuracy':accuracy,
		'precision':precision.item(),
		'recall':recall.item(),
		'fscore':fscore.item(),
		'confusion':confusion
	}
	
	return history
