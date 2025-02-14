import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from .models import merge_multiclass_mlps, classifier_evaluation
from .data import enforce_res


###! no straightforward way to make ART deterministic
###! currently, should set global seed to control randomness

###! enforce_res is currently mandatory in training functions


"""
type: (
	tf.keras.Model, 
	attacks.AbstractAttack, 
	np.ndarray, 
	np.ndarray, 
	np.ndarray, 
	np.ndarray, 
	np.ndarray,
	np.ndarray,
	np.ndarray,
	float,
	float,
	int,
	int,
	List[tf.keras.callbacks.Callback],
	bool
) -> Dict[str : List[Dict[str : {float, np.ndarray}]]]
"""
def uniform_adversarial_train(
		model,
		attack,
		feature_res,
		train_x,
		train_y,
		train_mask,
		val_x,
		val_y,
		val_mask,
		unif_lower=0.5,
		unif_upper=1.0,
		epochs=10,
		batch_size=64,
		callbacks=None,
		verbose=False
	):
	
	# assertions
	assert model.optimizer is not None and model.loss is not None and model.metrics is not None, "model must be precompiled"
	
	# init history
	history = {
		'train':[],
		'train_adv':[],
		'val':[],
		'val_adv':[],
	}
	
	# train model
	with tqdm(range(epochs), desc='Uniform PGD Train', unit='epoch') as bar:
		for i in bar:
			
			# generate adversarial samples with uniform-randomly scaled perturbations
			train_adv_xres = enforce_res(attack.generate(train_x, train_y, mask=train_mask), feature_res) - train_x ###! ND seed
			train_adv_x = train_x + train_adv_xres * np.random.uniform(unif_lower, unif_upper, (train_x.shape[0], 1)) ###! ND seed
			val_adv_x = enforce_res(attack.generate(val_x, val_y, mask=val_mask), feature_res) # no scaling on val ###! ND seed
			if verbose:
				print(f'[Elapsed time: {time.time()-T0:.2f}s]')
			
			# fit to augmented training set
			train_aug_x = np.concatenate([train_x, train_adv_x])
			train_aug_y = np.concatenate([train_y, train_y])
			train_history = model.fit(
				train_aug_x,
				train_aug_y,
				epochs=1,
				batch_size=batch_size,
				callbacks=callbacks,
				verbose=int(verbose)
			).history ###! ND seed
			train_history = {k:train_history[k][0] for k in train_history.keys()}
			
			# evaluate model
			val_history = classifier_evaluation(model, model.loss, val_x, val_y, batch_size=batch_size, verbose=verbose)
			val_adv_history = classifier_evaluation(model, model.loss, val_adv_x, val_y, batch_size=batch_size, verbose=verbose)
			train_adv_history = classifier_evaluation(model, model.loss, train_adv_x, train_y, batch_size=batch_size, verbose=verbose)
			
			# record results
			history['train'].append(train_history)
			history['val'].append(val_history)
			history['val_adv'].append(val_adv_history)
			history['train_adv'].append(train_adv_history)
			
			# update tqdm
			train_loss = train_history['loss']
			train_acc = train_history['accuracy']
			train_adv_acc = train_adv_history['accuracy']
			val_loss = val_history['loss']
			val_acc = val_history['accuracy']
			val_adv_acc = val_adv_history['accuracy']
			bar.set_postfix(
				loss=f'{train_loss:.4f}',
				accuracy=f'{train_acc:.4f}',
				adv_accuracy=f'{train_adv_acc:.4f}',
				val_loss=f'{val_loss:.4f}',
				val_accuracy=f'{val_acc:.4f}',
				val_adv_accuracy=f'{val_adv_acc:.4f}'
			)
	
	return history


"""
type: (
	tf.keras.Model,
	attacks.AbstractAttack,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	float,
	float,
	int,
	int,
	List[tf.keras.callbacks.Callback],
	bool
) -> Dict[str : List[Dict[str : {float, np.ndarray}]]]
"""
def scheduled_adversarial_train(
		model,
		attack,
		feature_res,
		train_x,
		train_y,
		train_mask,
		val_x,
		val_y,
		val_mask,
		schedule_start=1.0,
		schedule_stop=0.0, # linear decay
		epochs=10,
		batch_size=64,
		callbacks=None,
		verbose=False
	):
	
	# assertions
	assert model.optimizer is not None and model.loss is not None and model.metrics is not None, "model must be precompiled"
	
	# init history
	history = {
		'train':[],
		'train_adv':[],
		'val':[],
		'val_adv':[],
	}
	
	# train model
	with tqdm(np.linspace(schedule_start, schedule_stop, num=epochs), desc='Scheduled PGD Train', unit='epoch') as bar:
		for i in bar:
			
			# generate adversarial samples with scaled perturbations
			train_adv_xres = enforce_res(attack.generate(train_x, train_y, mask=train_mask), feature_res) - train_x ###! ND seed
			train_adv_x = train_x + train_adv_xres * i
			val_adv_x = enforce_res(attack.generate(val_x, val_y, mask=val_mask), feature_res) # no scaling on val ###! ND seed
			if verbose:
				print(f'[Elapsed time: {time.time()-T0:.2f}s]')
			
			# fit to augmented training set
			train_aug_x = np.concatenate([train_x, train_adv_x])
			train_aug_y = np.concatenate([train_y, train_y])
			train_history = model.fit(
				train_aug_x,
				train_aug_y,
				epochs=1,
				batch_size=batch_size,
				callbacks=callbacks,
				verbose=int(verbose)
			).history ###! ND seed
			train_history = {k:train_history[k][0] for k in train_history.keys()}
			
			# evaluate model
			val_history = classifier_evaluation(model, model.loss, val_x, val_y, batch_size=batch_size, verbose=verbose)
			val_adv_history = classifier_evaluation(model, model.loss, val_adv_x, val_y, batch_size=batch_size, verbose=verbose)
			train_adv_history = classifier_evaluation(model, model.loss, train_adv_x, train_y, batch_size=batch_size, verbose=verbose)
			
			# record results
			history['train'].append(train_history)
			history['val'].append(val_history)
			history['val_adv'].append(val_adv_history)
			history['train_adv'].append(train_adv_history)
			
			# update tqdm
			train_loss = train_history['loss']
			train_acc = train_history['accuracy']
			train_adv_acc = train_adv_history['accuracy']
			val_loss = val_history['loss']
			val_acc = val_history['accuracy']
			val_adv_acc = val_adv_history['accuracy']
			bar.set_postfix(
				loss=f'{train_loss:.4f}',
				accuracy=f'{train_acc:.4f}',
				adv_accuracy=f'{train_adv_acc:.4f}',
				val_loss=f'{val_loss:.4f}',
				val_accuracy=f'{val_acc:.4f}',
				val_adv_accuracy=f'{val_adv_acc:.4f}'
			)
	
	return history


"""
type: (
	lambda () -> tf.keras.Model, 
	lambda () -> tf.keras.losses.Loss, 
	lambda () -> tf.keras.optimizers.Optimizer, 
	lambda () -> List[str]
	np.ndarray, 
	np.ndarray, 
	np.ndarray, 
	np.ndarray,
	int,
	int,
	int,
	List[tf.keras.callbacks.Callback],
	bool
) -> Tuple(Dict, tf.keras.Model)
"""
def federated_train(
		model_init,
		criterion_init,
		optimizer_init,
		metrics_init,
		train_x,
		train_y,
		val_x,
		val_y,
		n_nodes=5,
		epochs=10,
		batch_size=64,
		callbacks=None,
		verbose=False
	):
	
	# initialise history
	history = {'nodes':[], 'node_weights':[]}
	
	# split data
	train_x_partitions = np.array_split(train_x, n_nodes)
	train_y_partitions = np.array_split(train_y, n_nodes)
	
	# initialise nodes
	nodes = []
	for i in range(n_nodes):
		node = model_init()
		node.name = f'federated_node_{i+1}'
		node.compile(loss=criterion_init(), optimizer=optimizer_init(), metrics=metrics_init())
		nodes.append(node)
	
	node_weights = [len(txp)/len(train_x) for txp in train_x_partitions]
	history['node_weights'] = node_weights
	
	# train nodes
	#for node, train_x_partition, train_y_partition in zip(nodes, train_x_partitions, train_y_partitions): # no tqdm
	train_zip = list(zip(nodes, train_x_partitions, train_y_partitions))
	for i in tqdm(range(len(train_zip)), desc='Federated train', unit='node'):
		node, train_x_partition, train_y_partition = train_zip[i] # tqdm workaround for progress bar
		node_history = node.fit(
			train_x_partition,
			train_y_partition,
			epochs=epochs,
			batch_size=batch_size,
			validation_data=(val_x, val_y),
			callbacks=callbacks,
			verbose=int(verbose)
		)
		history['nodes'].append(node_history.history)
	
	# merge nodes
	federated_model = merge_multiclass_mlps(nodes, weights=node_weights)
	
	return history, federated_model


"""
type: (
	lambda () -> tf.keras.Model, 
	lambda () -> tf.keras.losses.Loss, 
	lambda () -> tf.keras.optimizers.Optimizer, 
	lambda () -> List[str],
	lambda (tf.keras.Model) -> AbstractAttack,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	int,
	float,
	float,
	int,
	int,
	List[tf.keras.callbacks.Callback],
	bool
) -> Tuple(Dict, tf.keras.Model)
"""
def federated_uniform_adversarial_train(
		model_init,
		criterion_init,
		optimizer_init,
		metrics_init,
		attack_init,
		feature_res,
		train_x,
		train_y,
		train_mask,
		val_x,
		val_y,
		val_mask,
		n_nodes=5,
		unif_lower=0.5,
		unif_upper=1.0,
		epochs=10,
		batch_size=64,
		callbacks=None,
		verbose=False
	):
	
	# initialise history
	history = {'nodes':[], 'node_weights':[]}
	
	# split data
	train_x_partitions = np.array_split(train_x, n_nodes)
	train_y_partitions = np.array_split(train_y, n_nodes)
	
	# initialise nodes
	nodes = []
	for i in range(n_nodes):
		node = model_init()
		node.name = f'federated_node_{i+1}'
		node.compile(loss=criterion_init(), optimizer=optimizer_init(), metrics=metrics_init())
		nodes.append(node)
	
	node_weights = [len(txp)/len(train_x) for txp in train_x_partitions]
	history['node_weights'] = node_weights
	
	# train nodes
	#for node, train_x_partition, train_y_partition in zip(nodes, train_x_partitions, train_y_partitions): # no tqdm
	train_zip = list(zip(nodes, train_x_partitions, train_y_partitions))
	for i in tqdm(range(len(train_zip)), desc='Federated train', unit='node'):
		node, train_x_partition, train_y_partition = train_zip[i] # tqdm workaround for progress bar
		node_history = uniform_adversarial_train(
			node,
			attack_init(node),
			feature_res,
			train_x,
			train_y,
			train_mask,
			val_x,
			val_y,
			val_mask,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks,
			verbose=verbose
		)
		history['nodes'].append(node_history)
	
	# merge nodes
	federated_model = merge_multiclass_mlps(nodes, weights=node_weights)
	
	return history, federated_model
