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

# x inR (n_features, )
# z inR (n_classes, )
def baseline_ids(params, x, a=jax.nn.relu):
	
	###! implement biases
	
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

# params inR (*)
# -> inR
def l1_norm(params):
	if isinstance(params, jnp.ndarray):
		return jnp.sum(jnp.abs(params))
	elif isinstance(params, (list, tuple)):
		return jnp.sum(jnp.array([l1_norm(item) for item in params]))
	return 0

# params inR (*)
# -> inR
def lp_norm(params, p):
	assert p > 1
	if isinstance(params, jnp.ndarray):
		return jnp.linalg.norm(params, ord=p)
	elif isinstance(params, (list, tuple)):
		return jnp.power(jnp.sum(jnp.array([lp_norm(branch, p)**p for branch in params])), 1./p)
	return 0.

# params inR (*)
# -> inZ
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

def mask_fn(x):
	mask = jnp.zeros(x.shape)
	mask.at[2:2+x[1]].set(1.)
	return mask

# # x, mask inR (*, n_features)
# def can_dlc_mask(x):
	# mask = jnp.zeros(x.shape)
	# for i,sample in enumerate(mask):
		# mask = mask.at[i][2:2+sample[1]].set(1)
	# return mask.astype('float32')

# x, x_gridclip inR (*, n_features)
def enforce_res(x, feature_scale, mask=None):
	#assert (jnp.min(x) >= 0.) and (jnp.max(x) <= 1.)
	x_residual = x - jnp.round(jnp.minimum(feature_scale, jnp.maximum(0., x * feature_scale))) / feature_scale
	if mask is not None:
		x_residual *= mask
	x_gridclip = x - x_residual
	return x_gridclip

# x, mask, adv_x inR (*, n_features)
# epsilon inR
# clip_values inR (2)
def runif_perturbation(key, x, epsilon, clip_values=(0,1), mask=None):
	runif = jax.random.uniform(key, x.shape, minval=-epsilon, maxval=epsilon)
	if mask is not None:
		runif *= mask
	adv_x = x + runif
	adv_x = jnp.clip(adv_x, min=clip_values[0], max=clip_values[1])
	return adv_x

# p inR (*)
# x, mask, adv_x inR (*, n_features)
# y inR (*, n_classes)
# epsilon inR
# n_iterations inZ+
def pgd_perturbation(model, p, x, y, epsilon, n_iterations, clip_values=(0,1), targeted=False, mask=None, minimal=False):
	loss, grad = jax.value_and_grad(lambda x, y : cce_loss(model(p, x), y))(x, y)
	xp = jnp.sign(grad)
	if targeted:
		xp *= -1 # untested
	if mask is not None:
		xp *= mask
	if minimal:
		minimal_mask = jnp.array(jnp.argmax(model(p, x), axis=-1)==jnp.argmax(y, axis=-1)).astype('float32') # mask once misclassified
		xp *= minimal_mask[:, jnp.newaxis]
	adv_x = x + (epsilon/n_iterations) * xp
	adv_x = jnp.clip(adv_x, min=clip_values[0], max=clip_values[1]) # won't leave unit Linf since +/-1*(epsilon/n_iterations)*n_iterations = +/-budget, but can leave 0-1
	return adv_x


##### keys and hyperparameters


# init RNG seeds
key = jax.random.PRNGKey(0)
k0, key = jax.random.split(key, 2) # data shuffle
k1, key = jax.random.split(key, 2) # init params
k2, key = jax.random.split(key, 2) # train batching
k3, key = jax.random.split(key, 2) # test batching
k4, key = jax.random.split(key, 2) # adversarial init

# data
RATIO_TRAIN = 0.7 # in [0..1]
RATIO_VAL   = 0.05 # in [0..1]
RATIO_TEST  = 1 - (RATIO_TRAIN+RATIO_VAL) # in [0..1]
D_FEATURES  = 10 # inZ+
D_LABELS    = 5 # inZ+
FEATURE_SCALE = jnp.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255]) # inZ+ (10)

# training
EPOCHS     = 5 # inZ+
BATCH_SIZE = 512 # inZ+
ETA        = 0.001 # inR
R_P        = 2 # inZ+
R_LAM      = 0.01 # in [0..1]

# attack
PGD_EPS = 1
PGD_ITER = 7


##### load data and compile


# trace
print('Loading data and compiling..')

# load and preprocess dataset
data = pd.read_csv('car_hacking_dataset/car_hacking_dataset.csv', header=None)

###! features and labels should be pandas for adv align

# features = data[range(D_FEATURES)]
# features = ((features - features.min()) / (features.max() - features.min())).to_numpy() # min-max normalise features
# labels = jnp.squeeze(jax.nn.one_hot(data[[10]].to_numpy(), D_LABELS), axis=-2) # one-hot encode labels
features = jax.random.uniform(k2, shape=(512, 10,), minval=0.0, maxval=1.0)
labels = jax.nn.softmax(jax.random.uniform(k2, shape=(512, 5,), minval=0.0, maxval=1.0))

# shuffle paired indices
assert len(features)==len(labels)
shuffle_ind = jax.random.permutation(k0, len(features))
features = features[shuffle_ind]
labels = labels[shuffle_ind]

# train, val, test split
train_x = features[:int(RATIO_TRAIN*len(features))]
train_y = labels[:int(RATIO_TRAIN*len(labels))]
#train_mask = jnp.array([mask_fn(jnp.array([123,2,4,8,0,0,0,0,0,0])) for x in train_x])
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
hevs_params = init_baseline_ids(k1, [D_FEATURES, 16, 16, 16, 16, D_LABELS])

# define lp loss
def lp_loss_fn(p, x, y):
	return cce_loss(model(p, x), y) + R_LAM * lp_norm(p, R_P) # must be scalar valued for autodiff cannot return predictions

# define hevs loss
hevs_params_flat, hevs_params_tree = tree_flatten(hevs_params)
hevs_params_flat_sizes = jnp.array([jnp.prod(jnp.array(pf.shape)) for pf in hevs_params_flat])
hevs_params_flat_sizes_cumsum = jnp.cumsum(jnp.concatenate([jnp.array([0]), hevs_params_flat_sizes[:-1]]))
hevs_params_flat_shapes = jnp.array([pf.shape for pf in hevs_params_flat])
del(hevs_params_flat)

def flat_hevs_loss_fn(p_flat_1d, x, y):
	p_flat = [p_flat_1d[start:start+stop].reshape(shape) for start,stop,shape in zip(hevs_params_flat_sizes_cumsum, hevs_params_flat_sizes, hevs_params_flat_shapes)]
	p = tree_unflatten(hevs_params_tree, p_flat)
	return cce_loss(model(p, x), y)

def hevs_loss_fn(p, x, y):
	p_flat, _ = tree_flatten(p)
	p_flat_1d = jnp.concatenate([pf.flatten() for pf in p_flat])
	hevs = jnp.linalg.eigvals(jax.hessian(flat_hevs_loss_fn)(p_flat_1d, x, y)).astype('float32')
	
	hl = R_LAM * jnp.mean(jnp.pow(hevs, R_P))
	print(hl)
	return cce_loss(model(p, x), y) + hl

# compile lp optimizer
lp_opt_state = opt.init(lp_params)
@jax.jit
def lp_optimizer_step(state, p, x, y):
	loss, grad = jax.value_and_grad(lp_loss_fn)(p, x, y)
	updates, state = opt.update(grad, state, p)
	return loss, state, optax.apply_updates(p, updates)

# compile hevs optimizer
hevs_opt_state = opt.init(hevs_params)
#@jax.jit
def hevs_optimizer_step(state, p, x, y):
	loss, grad = jax.value_and_grad(hevs_loss_fn)(p, x, y)
	updates, state = opt.update(grad, state, p)
	return loss, state, optax.apply_updates(p, updates)

# compile attack
@jax.jit
def pgd_attack(key, p, x, y, targeted=False, mask=None, minimal=False):
	adv_x = runif_perturbation(key, x, PGD_EPS, mask=mask)
	adv_y = y #model(params, x) - 1e-4
	for i in range(PGD_ITER):
		adv_x = enforce_res(pgd_perturbation(model, p, adv_x, adv_y, PGD_EPS, PGD_ITER, targeted=targeted, mask=mask, minimal=minimal), FEATURE_SCALE)
	return adv_x

# trace
print(train_x.shape, train_y.shape, 'train shape')
print(val_x.shape, val_y.shape, 'validation shape')
print(test_x.shape, test_y.shape, 'test shape')
print(f'{count_params(lp_params)} model parameters')


##### train and evaluate Lp model


# init history
lp_history = {
	'train':{'loss':[], 'accuracy':[]},
	'val':{'loss':[], 'accuracy':[]},
	'test':{'loss':None, 'accuracy':None},
	'pgd':{'loss':None, 'accuracy':None},
	'params':None
}

# train model
checkpoint_loss = float('inf')
with tqdm(range(EPOCHS), desc='Lp Train', unit='epoch') as bar:
	for i in bar:
		
		# split k2
		k2s, k2 = jax.random.split(k2, 2)
		
		# train on batches
		train_batched = batch(k2s, train_x, train_y, BATCH_SIZE)
		epoch_loss = 0.0
		epoch_accuracy = 0.0
		for x, y in train_batched:
			loss, lp_opt_state, lp_params = lp_optimizer_step(lp_opt_state, lp_params, x, y)
			epoch_loss += loss
			epoch_accuracy += accuracy_score(model(lp_params, x), y) # recompute predictions
		
		# score means
		train_loss = epoch_loss / len(train_batched)
		train_accuracy = epoch_accuracy / len(train_batched)
		
		# evalulate on validation
		val_loss = lp_loss_fn(lp_params, val_x, val_y)
		val_accuracy = accuracy_score(model(lp_params, val_x), val_y) # recompute predictions
		
		# save best checkpoint
		if val_loss < checkpoint_loss:
			lp_history['params'] = lp_params # pointer or copy?
		
		# record
		lp_history['train']['loss'].append(train_loss)
		lp_history['train']['accuracy'].append(train_accuracy)
		lp_history['val']['loss'].append(val_loss)
		lp_history['val']['accuracy'].append(val_accuracy)
		
		bar.set_postfix(
			loss=f'{train_loss:.4f}',
			val_loss=f'{val_loss:.4f}',
		)

# reload best checkpoint
lp_params = lp_history['params']

# baseline evaluation
test_batched = batch(k3, test_x, test_y, BATCH_SIZE)
test_loss = jnp.mean(jnp.array([lp_loss_fn(lp_params, x, y) for x,y in test_batched]))
test_accuracy = jnp.mean(jnp.array([accuracy_score(model(lp_params, x), y) for x,y in test_batched])) # recompute predictions

# adversarial evaluation
test_batched_pgd = [(pgd_attack(k4s, lp_params, x, y), y) for k4s,(x,y) in zip(jax.random.split(k4, len(test_batched)), test_batched)]
test_pgd_loss = jnp.mean(jnp.array([lp_loss_fn(lp_params, adv_x, y) for adv_x,y in test_batched_pgd]))
test_pgd_accuracy = jnp.mean(jnp.array([accuracy_score(model(lp_params, adv_x), y) for adv_x,y in test_batched_pgd])) # recompute predictions

# record
lp_history['test']['loss'] = test_loss
lp_history['test']['accuracy'] = test_accuracy
lp_history['pgd']['loss'] = test_pgd_loss
lp_history['pgd']['accuracy'] = test_pgd_accuracy
lp_history['params'] = lp_params

# trace
print(f'test_loss={test_loss:.4f}, test_accuracy={test_accuracy:.4f}, test_pgd_loss={test_pgd_loss:.4f}, test_pgd_accuracy={test_pgd_accuracy:.4f}')
print('loss, val_loss: ' + ', '.join([f'[{float(tl):.4f}, {float(vl):.4f}]' for tl, vl in zip(lp_history['train']['loss'], lp_history['val']['loss'])]))
print('accuracy, val_accuracy: ' + ', '.join([f'[{float(tl):.4f}, {float(vl):.4f}]' for tl, vl in zip(lp_history['train']['accuracy'], lp_history['val']['accuracy'])]))
# test_loss=0.1117, test_accuracy=0.9634, test_pgd_loss=19.7653, test_pgd_accuracy=0.0029
# ['(0.1338, 0.1141)', '(0.1135, 0.1130)', '(0.1126, 0.1119)', '(0.1120, 0.1113)', '(0.1117, 0.1114)']

# save history
with open('bids_lp_history.pkl', 'wb') as f:
	pickle.dump(lp_history, f)


##### train and evaluate hevs model


# init history
hevs_history = {
	'train':{'loss':[], 'accuracy':[]},
	'val':{'loss':[], 'accuracy':[]},
	'test':{'loss':None, 'accuracy':None},
	'pgd':{'loss':None, 'accuracy':None},
	'params':None
}

# train model
checkpoint_loss = float('inf')
with tqdm(range(EPOCHS), desc='HEVs Train', unit='epoch') as bar:
	for i in bar:
		
		# split k2
		k2s, k2 = jax.random.split(k2, 2)
		
		# train on batches
		train_batched = batch(k2s, train_x, train_y, BATCH_SIZE)
		epoch_loss = 0.0
		epoch_accuracy = 0.0
		for x, y in train_batched:
			loss, hevs_opt_state, hevs_params = hevs_optimizer_step(hevs_opt_state, hevs_params, x, y)
			epoch_loss += loss
			epoch_accuracy += accuracy_score(model(hevs_params, x), y) # recompute predictions
		
		# score means
		train_loss = epoch_loss / len(train_batched)
		train_accuracy = epoch_accuracy / len(train_batched)
		
		# evalulate on validation
		val_loss = hevs_loss_fn(hevs_params, val_x, val_y)
		val_accuracy = accuracy_score(model(hevs_params, val_x), val_y) # recompute predictions
		
		# save best checkpoint
		if val_loss < checkpoint_loss:
			hevs_history['params'] = hevs_params # pointer or copy?
		
		# record
		hevs_history['train']['loss'].append(train_loss)
		hevs_history['train']['accuracy'].append(train_accuracy)
		hevs_history['val']['loss'].append(val_loss)
		hevs_history['val']['accuracy'].append(val_accuracy)
		
		bar.set_postfix(
			loss=f'{train_loss:.4f}',
			val_loss=f'{val_loss:.4f}',
		)

# reload best checkpoint
hevs_params = hevs_history['params']

# baseline evaluation
test_batched = batch(k3, test_x, test_y, BATCH_SIZE)
test_loss = jnp.mean(jnp.array([hevs_loss_fn(hevs_params, x, y) for x,y in test_batched]))
test_accuracy = jnp.mean(jnp.array([accuracy_score(model(hevs_params, x), y) for x,y in test_batched])) # recompute predictions

# adversarial evaluation
test_batched_pgd = [(pgd_attack(k4s, hevs_params, x, y), y) for k4s,(x,y) in zip(jax.random.split(k4, len(test_batched)), test_batched)]
test_pgd_loss = jnp.mean(jnp.array([hevs_loss_fn(hevs_params, adv_x, y) for adv_x,y in test_batched_pgd]))
test_pgd_accuracy = jnp.mean(jnp.array([accuracy_score(model(hevs_params, adv_x), y) for adv_x,y in test_batched_pgd])) # recompute predictions

# record
hevs_history['test']['loss'] = test_loss
hevs_history['test']['accuracy'] = test_accuracy
hevs_history['pgd']['loss'] = test_pgd_loss
hevs_history['pgd']['accuracy'] = test_pgd_accuracy
hevs_history['params'] = hevs_params

# trace
print(f'test_loss={test_loss:.4f}, test_accuracy={test_accuracy:.4f}, test_pgd_loss={test_pgd_loss:.4f}, test_pgd_accuracy={test_pgd_accuracy:.4f}')
print('loss, val_loss: ' + ', '.join([f'[{float(tl):.4f}, {float(vl):.4f}]' for tl, vl in zip(hevs_history['train']['loss'], hevs_history['val']['loss'])]))
print('accuracy, val_accuracy: ' + ', '.join([f'[{float(tl):.4f}, {float(vl):.4f}]' for tl, vl in zip(hevs_history['train']['accuracy'], hevs_history['val']['accuracy'])]))

# save history
with open('bids_hevs_history.pkl', 'wb') as f:
	pickle.dump(hevs_history, f)
