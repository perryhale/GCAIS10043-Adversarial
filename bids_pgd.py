"""Projected Gradient Descent"""
import os; os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
import jax
import jax.numpy as jnp
import pandas as pd
import pickle
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt; plt.clf() # patch

# x inR (input_dim, )
# z inR (output_dim, )
def baseline_ids(params, x, a=jax.nn.relu):
	z = x
	for w in params:
		z = a(jnp.dot(w, z))
	z = jax.nn.softmax(z)
	return z

# yh, y inR (*, n_classes)
# loss inR
def cce_loss(yh, y, e=1e-9):
	loss = -jnp.mean(jnp.sum(y * jnp.log(yh + e), axis=-1))
	return loss

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
	accuracy = jnp.mean(jnp.array(yhc==yc, dtype='float32'))
	return accuracy



# initialise RNG seed
key = jax.random.PRNGKey(0)



# load dataset
k0, key = jax.random.split(key, 2)

data = pd.read_csv('car_hacking_dataset/car_hacking_dataset.csv', header=None)
features = data[[0,1,2,3,4,5,6,7,8,9]]
features = ((features - features.min()) / (features.max() - features.min())).to_numpy() # column-wise min-max norm
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
with open('models/baseline_ids_jax.weights.pkl', 'rb') as f:
	params = pickle.load(f)
model = jax.vmap(baseline_ids, in_axes=(None, 0))

print(f'{count_params(params)} parameters')



# initialise adversarial optimizer
epsilon = 0.25
iterations = 7

adv_loss_fn = lambda xs,p,ys : cce_loss(model(p, xs), ys)
adv_minimal = False
adv_mask = jnp.array([0.,0.,1.,1.,1.,1.,1.,1.,1.,1.]).reshape((10,))

@jax.jit
def enforce_255(xs): # enforce 8bit resolution (0-255 discrete)
	xs_slice = xs[:, 2:]
	xs_slice = jnp.minimum(jnp.maximum(xs_slice * 255.0, 0.0), 255.0).astype(jnp.int32) # floor op
	#print(xs_slice[:8])
	xs_slice = xs_slice.astype(jnp.float32) / 255.0
	xsh = xs.at[:, 2:].set(xs_slice)
	return xsh

@jax.jit
def runif_perturbation(key, xs):
	runif = jax.random.uniform(key, xs.shape, minval=-epsilon, maxval=epsilon)
	if adv_mask is not None:
		runif *= adv_mask
	return xs + runif

@jax.jit
def pgd_perturbation(xs, params, ys):
	loss, grad = jax.value_and_grad(adv_loss_fn)(xs, params, ys)
	ptbs = jnp.sign(grad)
	
	if adv_minimal:
		minimal_mask = jnp.array(jnp.argmax(model(params, xs), axis=-1)==jnp.argmax(ys, axis=-1), dtype='float32')
		ptbs *= minimal_mask[:, jnp.newaxis]
		#print(minimal_mask)
	
	if adv_mask is not None:
		ptbs *= adv_mask
	
	adv_xs = xs + (epsilon/iterations) * ptbs
	#adv_xs = jnp.clip(adv_xs, min=adv_xs-epsilon, max=adv_xs+epsilon) # redundant since +/-1*(eps/iter)*iter = +/-eps
	return loss, adv_xs



# generate adversarial samples
k1, key = jax.random.split(key, 2)

adv_x = runif_perturbation(k1, val_x)
adv_y = val_y
#adv_y = model(params, val_x) - 1e-4

history = {'loss':[], 'accuracy':[], 'similarity':[]}

for i in tqdm(range(iterations)):
	loss, adv_x = pgd_perturbation(adv_x, params, adv_y)
	adv_x = enforce_255(adv_x)
	#print(adv_x[:8])
	
	history['loss'].append(loss)
	history['accuracy'].append(accuracy_score(model(params, adv_x), adv_y))
	history['similarity'].append(jnp.mean(jnp.array([1 - scipy.spatial.distance.cosine(a,b) for a,b in zip(adv_x, val_x)])))

print([f'{float(l):.2f}' for l in history['loss']])
print(f'Baseline accuracy: {accuracy_score(model(params, val_x), val_y)}')
print(f'Adversarial accuracy: {accuracy_score(model(params, adv_x), adv_y)}')



# plot results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10,3))
ax1.plot(history['loss'])
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Loss')
ax2.plot(history['accuracy'], c='green')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Accuracy')
ax3.plot(history['similarity'], c='red')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Cosine similarity')
plt.tight_layout()
plt.show()
