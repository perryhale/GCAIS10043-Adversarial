# system
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
os.environ["G_MESSAGES_DEBUG"] = ""

# model
import jax
import jax.numpy as jnp
import optax
import pickle

# data
import pandas as pd

# util
from tqdm import tqdm
import matplotlib.pyplot as plt
from jax.tree_util import tree_flatten, tree_unflatten


##### functions


# shape inZ+ (*)
# fan_in, fan_out inZ+
# w inR shape
def init_glorot_uniform(key, shape, fan_in, fan_out):
	limit = jnp.sqrt(6) / jnp.sqrt(fan_in+fan_out)
	w = jax.random.uniform(key, shape, minval=-limit, maxval=limit)
	return w

# x inR (input_dim, )
# z inR (output_dim, )
def baseline_ids(params, x, a=jax.nn.relu):
	
	# implement biases
	
	z = x
	for w in params:
		z = a(jnp.dot(w, z))
	z = jax.nn.softmax(z)
	return z

# layers inZ+ []
def init_baseline_ids(key, layers):
	k0, key = jax.random.split(key)
	ws = [init_glorot_uniform(k0, (layers[1], layers[0]), layers[0], layers[1])]
	for i in range(1, len(layers)-1):
		k1, key = jax.random.split(key)
		ws.append(init_glorot_uniform(k1, (layers[i+1], layers[i]), layers[i], layers[i+1]))
	return ws

# xs, ys inR (samples, *)
# z inR (batches, 2, batch_size, *)
def batch(key, xs, ys, batch_size):
	assert len(xs)==len(ys)
	ind = jax.random.permutation(key, len(xs))
	sh_xs = xs[ind]
	sh_ys = ys[ind]
	ba_xs = [sh_xs[k:k+batch_size] for k in range(0, len(sh_xs), batch_size)]
	ba_ys = [sh_ys[k:k+batch_size] for k in range(0, len(sh_ys), batch_size)]
	return list(zip(ba_xs, ba_ys))

# yh, y inR (*, n_classes)
# loss inR
def cce_loss(yh, y, e=1e-9):
	loss = -jnp.mean(jnp.sum(y * jnp.log(yh + e), axis=-1))
	return loss

# ~
def l1_norm(params):
	if isinstance(params, jnp.ndarray):
		return jnp.sum(jnp.abs(params))
	elif isinstance(params, (list, tuple)):
		return jnp.sum(jnp.array([l1_norm(item) for item in params]))
	return 0

# ~
def lp_norm(params, p):
	assert p > 1
	if isinstance(params, jnp.ndarray):
		return jnp.linalg.norm(params, ord=p)
	elif isinstance(params, (list, tuple)):
		return jnp.power(jnp.sum(jnp.array([lp_norm(branch, p)**p for branch in params])), 1./p)
	return 0.

# ~
def count_params(params):
	if isinstance(params, jnp.ndarray):
		return jnp.prod(jnp.array(params.shape))
	elif isinstance(params, (list, tuple)):
		return jnp.sum(jnp.array([count_params(item) for item in params]))
	return 0

# yh, y inR (*, n_classes)
# accuracy inR
def accuracy_score(yh, y):
	yhc = jnp.argmax(yh, axis=-1)
	yc = jnp.argmax(y, axis=-1)
	accuracy = jnp.mean(jnp.array(yhc==yc, dtype='int32'))
	return accuracy













# def mask_fn(x):
	# mask = np.zeros(x.shape)
	# mask[2:2+x[1]] = 1.
	# return mask

def dlc_mask(xs):
	pass

def enforce_res(xs, res, mask=None):
	res = xs - np.round(np.minimum(res, np.maximum(0., xs * res))) / res
	if mask is not None:
		res *= mask
	return xs - res

def runif_perturbation(key, xs, budget, mask=None):
	runif = jax.random.uniform(key, xs.shape, minval=-budget, maxval=budget)
	if mask is not None:
		runif *= mask
	return xs + runif

def pgd_perturbation(xs, ys, model, params, budget, n_iter, mask=None, minimal=False, targeted=False):
	
	# get signed gradient
	loss, grad = jax.value_and_grad(lambda xs,ys : cce_loss(model(params, xs), ys))(xs, ys)
	ptbs = jnp.sign(grad)
	if targeted: ptbs *= -1 # untested
	
	# (optional) apply sample mask
	if mask is not None:
		ptbs *= mask
	
	# (optional) mask once misclassified
	if minimal:
		minimal_mask = jnp.array(jnp.argmax(model(params, xs), axis=-1)==jnp.argmax(ys, axis=-1), dtype='float32') # mask once misclassified
		ptbs *= minimal_mask[:, jnp.newaxis]
	
	# apply perturbation
	adv_xs = xs + (budget/n_iter) * ptbs
	adv_xs = jnp.clip(adv_xs, min=adv_xs-budget, max=adv_xs+budget) # redundant when +/-1*(budget/n_iter)*n_iter = +/-budget, but not when runif init
	
	return loss, adv_xs




















##### keys and hyperparameters


# init RNG seeds
key = jax.random.PRNGKey(0)
k0, key = jax.random.split(key, 2) # data shuffle
k1, key = jax.random.split(key, 2) # init params
k2, key = jax.random.split(key, 2) # train batching
k3, key = jax.random.split(key, 2) # test batching

# data
RATIO_TRAIN = 0.7 # in [0..1]
RATIO_VAL   = 0.05 # in [0..1]
RATIO_TEST  = 1 - (RATIO_TRAIN+RATIO_VAL) # in [0..1]
D_FEATURES  = 10 # inZ+
D_LABELS    = 5 # inZ+

# training
EPOCHS     = 5 # inZ+
BATCH_SIZE = 512 # inZ+
ETA        = 0.001 # inR
R_P         = 2 # inZ+
R_LAM     = 0.0001 # in [0..1]


##### load data and compile


# trace
print('Loading data and compiling..')

# load and preprocess dataset
#data = pd.read_csv('car_hacking_dataset/car_hacking_dataset.csv', header=None)

###! features and labels should be pandas for adv align

#features = data[range(D_FEATURES)]
#features = ((features - features.min()) / (features.max() - features.min())).to_numpy() # min-max normalise features
#labels = jnp.squeeze(jax.nn.one_hot(data[[10]].to_numpy(), D_LABELS), axis=-2) # one-hot encode labels
features = jax.random.normal(k2, shape=(512, 10,))
labels = jax.random.normal(k2, shape=(512, 5,))

# shuffle paired indices
assert len(features)==len(labels)
shuffle_ind = jax.random.permutation(k0, len(features))
features = features[shuffle_ind]
labels = labels[shuffle_ind]

# train, val, test split
train_x = features[:int(RATIO_TRAIN*len(features))]
train_y = labels[:int(RATIO_TRAIN*len(labels))]
val_x = features[int(RATIO_TRAIN*len(features)):int((RATIO_TRAIN+RATIO_VAL)*len(features))]
val_y = labels[int(RATIO_TRAIN*len(labels)):int((RATIO_TRAIN+RATIO_VAL)*len(labels))]
test_x = features[int((RATIO_TRAIN+RATIO_VAL)*len(features)):]
test_y = labels[int((RATIO_TRAIN+RATIO_VAL)*len(labels)):]

# oversampling train set
#train_x, train_y = RandomOverSampler().fit_resample(train_x, train_y)

# init model
model = jax.vmap(baseline_ids, in_axes=(None, 0))
opt = optax.adamw(ETA)

# init parameters
lp_params = init_baseline_ids(k1, [D_FEATURES, 16, 16, 16, 16, D_LABELS])
hp_params = init_baseline_ids(k1, [D_FEATURES, 16, 16, 16, 16, D_LABELS])

# define lp loss
def lp_loss_fn(p, x, y):
	return cce_loss(model(p, x), y) + R_LAM * lp_norm(p, R_P) # must be scalar valued cannot return predictions

# define hp loss
hp_params_flat, hp_params_tree = tree_flatten(hp_params)
hp_params_flat_sizes = jnp.array([jnp.prod(jnp.array(pf.shape)) for pf in hp_params_flat])
hp_params_flat_sizes_cumsum = jnp.cumsum(jnp.concatenate([jnp.array([0]), hp_params_flat_sizes[:-1]]))
hp_params_flat_shapes = jnp.array([pf.shape for pf in hp_params_flat])
del(hp_params_flat)

def flat_hp_loss_fn(p_flat_1d, x, y):
	p_flat = [p_flat_1d[start:start+stop].reshape(shape) for start,stop,shape in zip(hp_params_flat_sizes_cumsum, hp_params_flat_sizes, hp_params_flat_shapes)]
	p = tree_unflatten(hp_params_tree, p_flat)
	return cce_loss(model(p, x), y)

def hp_loss_fn(p, x, y):
	p_flat, _ = tree_flatten(p)
	p_flat_1d = jnp.concatenate([pf.flatten() for pf in p_flat])
	hevs = jnp.linalg.eigvals(jax.hessian(flat_hp_loss_fn)(p_flat_1d, x, y)).astype('float32')
	return cce_loss(model(p, x), y) + R_LAM * jnp.mean(jnp.pow(hevs, R_P))

# compile optimizers
lp_opt_state = opt.init(lp_params)
@jax.jit
def lp_optimizer_step(state, p, x, y):
	loss, grad = jax.value_and_grad(lp_loss_fn)(p, x, y)
	updates, state = opt.update(grad, state, p)
	return loss, state, optax.apply_updates(p, updates)

hp_opt_state = opt.init(hp_params)
@jax.jit
def hp_optimizer_step(state, p, x, y):
	loss, grad = jax.value_and_grad(hp_loss_fn)(p, x, y)
	updates, state = opt.update(grad, state, p)
	return loss, state, optax.apply_updates(p, updates)

# compile attack
# @jax.jit
# def attack(key, xs, ys):
	
	# # init
	# k0, key = jax.random.split(key, 2)
	# adv_xs = runif_perturbation(k0, xs)
	# adv_ys = ys
	# #adv_y = model(params, val_x) - 1e-4
	
	# # pgd iterations
	# history = {'loss':[], 'accuracy':[], 'distance':[]}
	# for i in tqdm(range(ITER)):
		
		# # iterate perturbation
		# loss, adv_xs = pgd_perturbation(adv_xs, adv_ys, MODEL, PARAMS, EPS, ITER)
		# adv_xs = enforce_255(adv_xs)
		# #print(adv_x[:8])
		
		# # evaluate
		# adv_accuracy = accuracy_score(MODEL(PARAMS, adv_xs), adv_ys)
		# adv_distance = jnp.mean(jnp.array([1 - scipy.spatial.distance.euclidean(a,b) for a,b in zip(adv_xs, xs)]))
		
		# # record
		# history['loss'].append(loss)
		# history['accuracy'].append(adv_accuracy)
		# history['distance'].append(adv_distance)
	
	# return history, (adv_xs, adv_ys)

# trace
print(train_x.shape, train_y.shape, 'train shape')
print(val_x.shape, val_y.shape, 'validation shape')
print(test_x.shape, test_y.shape, 'test shape')
print(f'{count_params(lp_params)} model parameters')


##### train model with Lp regularizer


# init history
history = {
	'train':{'loss':[], 'accuracy':[]},
	'val':{'loss':[], 'accuracy':[]},
	'test':{'loss':None, 'accuracy':None}
}

# train model
with tqdm(range(EPOCHS), desc='Train', unit='epoch') as bar:
	for i in bar:
		
		# split k2
		k2s, k2 = jax.random.split(k2, 2)
		
		# train on batches
		train_batched = batch(k2s, train_x, train_y, BATCH_SIZE)
		epoch_loss = 0.0
		epoch_accuracy = 0.0
		for x, y in train_batched:
			loss, hp_opt_state, hp_params = hp_optimizer_step(hp_opt_state, hp_params, x, y)
			epoch_loss += loss
			epoch_accuracy += accuracy_score(model(hp_params, x), y) # recompute predictions
		
		# score means
		train_loss = epoch_loss / len(train_batched)
		train_accuracy = epoch_accuracy / len(train_batched)
		
		# evalulate on validation
		val_loss = hp_loss_fn(hp_params, val_x, val_y)
		val_accuracy = accuracy_score(model(hp_params, val_x), val_y) # recompute predictions
		
		# record
		history['train']['loss'].append(train_loss)
		history['train']['accuracy'].append(train_accuracy)
		history['val']['loss'].append(val_loss)
		history['val']['accuracy'].append(val_accuracy)
		
		bar.set_postfix(
			loss=f'{train_loss:.4f}',
			val_loss=f'{val_loss:.4f}',
		)

# evaluate on test
test_batched = batch(k3, test_x, test_y, BATCH_SIZE)
test_loss = jnp.mean(jnp.array([hp_loss_function(hp_params, x, y) for x,y in test_batched]))
test_accuracy = jnp.mean(jnp.array([accuracy_score(model(hp_params, x), y) for x,y in test_batched])) # recompute predictions

# record
history['test']['loss'] = test_loss
history['test']['accuracy'] = test_accuracy

# trace
print(f'test_loss={test_loss:.8f}, test_accuracy={test_accuracy:.8f}')
print([f'({float(tl):.4f}, {float(vl):.4f})' for tl, vl in zip(history['train']['loss'], history['val']['loss'])])


##### evaluate Lp model on adversarial attack

##### train model with complexity regularizer

##### evaluate complexity model on adversarial attack 

##### format results

import sys;sys.exit()





