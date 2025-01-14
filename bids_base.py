import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
import jax
import jax.numpy as jnp
import optax
import pickle
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt; plt.clf() # patch

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





# initialise RNG seed
key = jax.random.PRNGKey(0)



# load dataset
k0, key = jax.random.split(key, 2)

data = pd.read_csv('car_hacking_dataset/car_hacking_dataset.csv', header=None)
features = data[[0,1,2,3,4,5,6,7,8,9]]
features = ((features - features.min()) / (features.max() - features.min())).to_numpy() # column-wise
labels = jnp.squeeze(jax.nn.one_hot(data[[10]].to_numpy(), 5), axis=-2)

assert len(features)==len(labels)
shuffle_ind = jax.random.permutation(k0, len(features))
features = features[shuffle_ind]
labels = labels[shuffle_ind]

val_x = features[int(0.7*len(features)):int(0.7*len(features))+500]
val_y = labels[int(0.7*len(labels)):int(0.7*len(labels))+500]
test_x = features[int(0.7*len(features))+500:]
test_y = labels[int(0.7*len(labels))+500:]
train_x = features[:int(0.7*len(features))]
train_y = labels[:int(0.7*len(labels))]

print(train_x.shape, train_y.shape, 'train')
print(val_x.shape, val_y.shape, 'validation')
print(test_x.shape, test_y.shape, 'test')



# initialise model
input_dim = 10
output_dim = 5
k0, key = jax.random.split(key, 2)

params = init_baseline_ids(k0, [input_dim, 16, 16, 16, 16, output_dim])
model = jax.vmap(baseline_ids, in_axes=(None, 0))

print(f'{count_params(params)} parameters')



# initialise optimizer and loss function
lr = 0.001 # inR
lam = 0.01 # in [0..1]

loss_fn = lambda pa,xs,ys : cce_loss(model(pa,xs),ys) + lam*lp_norm(pa,2)
opt = optax.adamw(lr)
state = opt.init(params)

@jax.jit
def optimizer_step(state, params, x, y):
	loss, grad = jax.value_and_grad(loss_fn)(params, x, y)
	updates, state = opt.update(grad, state, params)
	return loss, state, optax.apply_updates(params, updates)



# training loop
epochs = 5
batch_size = 512

# xs, ys inR (samples, *)
# z inR (batches, 2, batch_size, *)
#@jax.jit
def batch(key, xs, ys):
	assert len(xs)==len(ys)
	ind = jax.random.permutation(key, len(xs))
	sh_xs = xs[ind]
	sh_ys = ys[ind]
	ba_xs = [sh_xs[k:k+batch_size] for k in range(0, len(sh_xs), batch_size)]
	ba_ys = [sh_ys[k:k+batch_size] for k in range(0, len(sh_ys), batch_size)]
	return list(zip(ba_xs, ba_ys))

history = {'loss':[], 'val_loss':[]}

for i in tqdm(range(epochs)):
	k1, key = jax.random.split(key, 2)
	train_batched = batch(k1, train_x, train_y)
	epoch_loss = 0.0
	for x, y in train_batched:
		loss, state, params = optimizer_step(state, params, x, y)
		epoch_loss += loss
	
	history['loss'].append(epoch_loss / len(train_batched))
	history['val_loss'].append(loss_fn(params, val_x, val_y))




# evaluate
k2, key = jax.random.split(key, 2)
test_batched = batch(k2, test_x, test_y) # GPU memory requirement

test_loss = jnp.mean(jnp.array([loss_fn(params, x, y) for x,y in test_batched]))
test_accuracy = jnp.mean(jnp.array([accuracy_score(model(params, x), y) for x,y in test_batched]))
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_accuracy}')
print([f'{float(l):.2f}' for l in history['loss']])



# plot results
plt.plot(history['loss'], label='train')
plt.plot(history['val_loss'], label='validation', c='r')
plt.scatter([epochs-1], [test_loss], marker='x', label='test', c='g')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()



# save parameters
with open('models/baseline_ids_jax.weights.pkl', 'wb') as f:
	pickle.dump(params, f)
