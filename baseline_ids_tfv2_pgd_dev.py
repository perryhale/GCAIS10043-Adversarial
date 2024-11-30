import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import ops
from tensorflow.keras import optimizers

import art
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import ProjectedGradientDescent

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics.pairwise import cosine_distances



# set RNG seed
RNG_SEED = 0
tf.keras.utils.set_random_seed(RNG_SEED)
np.random.seed(RNG_SEED)



# load dataset
FEATURE_SCALE = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255])

def mask_fn(x):
	mask = np.zeros(x.shape)
	mask[2:2+x[1]] = 1.
	# if x[1] < 8:
		# print(x)
		# print(mask)
	return mask

data = pd.read_csv('car_hacking_dataset/car_hacking_dataset.csv', header=None)
data = data.sample(frac=1)[:10_000]
data_ben = data.loc[data[10] == 0]
data_mal = data.loc[data[10] >= 1]

ben_x = data_ben.iloc[:, :-1]
ben_x = ben_x.to_numpy() / FEATURE_SCALE
ben_y = data_ben.iloc[:, -1]
ben_y = ben_y.to_numpy() # true labels

mal_x = data_mal.iloc[:, :-1]
mal_x = mal_x.to_numpy() / FEATURE_SCALE
mal_y = data_mal.iloc[:, -1]
mal_y = mal_y.to_numpy() # true labels
mal_yt = np.zeros(len(mal_y)) # false labels

ben_mask = np.apply_along_axis(mask_fn, axis=1, arr=data_ben.iloc[:, :-1].to_numpy())
mal_mask = np.apply_along_axis(mask_fn, axis=1, arr=data_mal.iloc[:, :-1].to_numpy())

print(ben_x.shape, ben_y.shape, 'benign set')
print(mal_x.shape, mal_y.shape, mal_yt.shape, 'malicious set')



# define model
model_x = layers.Input(shape=(10,), name='baseline_model_input')
model_y = layers.Dense(16, activation='relu', name='baseline_model_hidden1')(model_x)
model_y = layers.Dense(16, activation='relu', name='baseline_model_hidden2')(model_y)
model_y = layers.Dense(16, activation='relu', name='baseline_model_hidden3')(model_y)
model_y = layers.Dense(16, activation='relu', name='baseline_model_hidden4')(model_y)
model_y = layers.Dense(5, activation='softmax', name='baseline_model_output')(model_y)

model = tf.keras.Model(model_x, model_y, name='baseline_ids')
model.load_weights('models/baseline_ids_tfv2.weights.h5')
model.summary()



# setup art wrapper
EPS = 16/255
ITER = 7

model = TensorFlowV2Classifier(model=model, nb_classes=5, input_shape=(10,), loss_object=tf.keras.losses.SparseCategoricalCrossentropy(), clip_values=(0,1))
pgd_untargeted = ProjectedGradientDescent(estimator=model, eps=EPS, eps_step=(EPS/ITER), max_iter=ITER, num_random_init=1, targeted=False)
pgd_targeted = ProjectedGradientDescent(estimator=model, eps=EPS, eps_step=(EPS/ITER), max_iter=ITER, num_random_init=1, targeted=True)



# generate samples
def enforce_res(xs, res, mask=None):
	res = xs - np.round(np.minimum(res, np.maximum(0., xs * res))) / res
	if mask is not None:
		res *= mask
	return xs - res

ben_adv_x = enforce_res(pgd_untargeted.generate(ben_x, mask=ben_mask), FEATURE_SCALE, mask=ben_mask)
mal_adv_x = enforce_res(pgd_targeted.generate(mal_x, mal_yt, mask=mal_mask), FEATURE_SCALE, mask=mal_mask)

print(ben_mask)
print(mal_mask)
print(ben_adv_x*FEATURE_SCALE)
print(mal_adv_x*FEATURE_SCALE)
# * epsion=64/255, ITER=7
# adversarial benign->malicious
# res=lower:	25.5329%
# res=0:		25.5097%
# res=round:	25.4981%
# mask=adaptive:23.2970%



# evaluate model on samples
ben_yh = np.argmax(model.predict(ben_x), axis=-1)
mal_yh = np.argmax(model.predict(mal_x), axis=-1)
ben_adv_yh = np.argmax(model.predict(ben_adv_x), axis=-1)
mal_adv_yh = np.argmax(model.predict(mal_adv_x), axis=-1)

print('----')
print(f'Baseline benign accuracy: {accuracy_score(ben_y, ben_yh)}')
print(f'Baseline malicious accuracy: {accuracy_score(mal_y, mal_yh)}')
print('----')
print(f'Adversarial benign accuracy: {accuracy_score(ben_adv_yh, ben_y)}')
print(f'Adversarial malicious accuracy: {accuracy_score(mal_adv_yh, mal_y)}')
print(f'Adversarial malicious targeted accuracy: {accuracy_score(mal_adv_yh, mal_yt)}')

ben_cfm = confusion_matrix(ben_y, ben_yh, labels=range(5)) # y,yh
mal_cfm = confusion_matrix(mal_y, mal_yh, labels=range(5))
ben_adv_cfm = confusion_matrix(ben_y, ben_adv_yh, labels=range(5))
mal_adv_cfm = confusion_matrix(mal_y, mal_adv_yh, labels=range(5))

print('----')
print(ben_cfm)
print(mal_cfm)
print('----')
print(ben_adv_cfm)
print(mal_adv_cfm)
# * EPS=64/255, ITER=7
# ----
# Baseline benign accuracy: 0.9964087117701576
# Baseline malicious accuracy: 0.97953216374269
# ----
# Adversarial benign accuracy: 0.23297034291010194
# Adversarial malicious accuracy: 0.1074561403508772
# Adversarial malicious targeted accuracy: 0.8925438596491229
# ----
# [[8601   21    3    6    1]
 # [   0    0    0    0    0]
 # [   0    0    0    0    0]
 # [   0    0    0    0    0]
 # [   0    0    0    0    0]]
# [[  0   0   0   0   0]
 # [  0 378   0   0   0]
 # [ 28   0 257   0   0]
 # [  0   0   0 341   0]
 # [  0   0   0   0 364]]
# ----
# [[2011  309 5872  176  264]
 # [   0    0    0    0    0]
 # [   0    0    0    0    0]
 # [   0    0    0    0    0]
 # [   0    0    0    0    0]]
# [[  0   0   0   0   0]
 # [378   0   0   0   0]
 # [139   0 146   0   0]
 # [341   0   0   0   0]
 # [363   0   0   0   1]]



# measure cosine similarities
cosim_ben_adv = 1 - cosine_distances(ben_x, ben_adv_x)
cosim_mal_adv = 1 - cosine_distances(mal_x, mal_adv_x)
cosim_mal_advt = 1 - cosine_distances(mal_adv_x, ben_x) # non-square

cosim_ben_adv_diag_mean = np.mean(np.diag(cosim_ben_adv))
cosim_mal_adv_diag_mean = np.mean(np.diag(cosim_mal_adv))
cosim_mal_advt_max_mean = np.mean(np.max(cosim_mal_advt, axis=1))

print('----')
print(f'ben_x, ben_adv_x cosine similarity diag mean: {cosim_ben_adv_diag_mean}')
print(f'mal_x, mal_adv_x cosine similarity diag mean: {cosim_mal_adv_diag_mean}')
print(f'mal_adv_x, ben_x cosine similarity max mean: {cosim_mal_advt_max_mean}')
# ----
# ben_x, ben_adv_x cosine similarity diag mean: 0.9293888171704988
# mal_x, mal_adv_x cosine similarity diag mean: 0.9532749021303012
# mal_adv_x, ben_x cosine similarity max mean: 0.9639887431823045






### notes



### 16:45 16th Nov 24
### data preprocessing method is wrong, model needs retraining with correct method
### each column needs different scale: 4095, 8, 255, 255, 255, 255, 255, 255, 255, 255
### use confusion matrices for evaluation

### 01:39 17th Nov 24
### mask should depend on dlc, would not work as a postprocess because the gradient would have changed depending on the illegal axes
### mask needs to be a function of the input but art mask is constant
### non-iterative attack would allow a valid postprocess solution, because the partial derivatives are independent

### 12:10 17th Nov 23
### could override _apply_perturbation of ProjectedGradientDescentTensorFlowV2, but this would also require a custom ProjectedGradientDescent

# from art.attacks.evasion import ProjectedGradientDescent, ProjectedGradientDescentTensorFlowV2

# class PGDAdaptiveMask(ProjectedGradientDescent):
	# def __init__(self, **kwargs):
		# super().__init__(**kwargs)
		# self._attack = PGDTFV2AdaptiveMask(**kwargs)
	
	# def generate(self, *args, mask_fn=None, **kwargs):
		# kwargs['mask_fn'] = mask_fn
		# return super().generate(*args, **kwargs)

# class PGDTFV2AdaptiveMask(ProjectedGradientDescentTensorFlowV2):
	# def __init__(self, **kwargs):
		# super().__init__(**kwargs)
	
	# def generate(self, *args, **kwargs):
		# mask_fn = kwargs.get('mask_fn', 'default_value')
		# return super().generate(*args, **kwargs)

# #mask_fn x : z
# #attack.generate(set_x, mask=feature_mask, mask_fn=mask_fn)

### 18:30 17th Nov
### sample-specific masks are supported by art so can generate masks for each sample



### preprocessing issue
# ----                                 
# Baseline regign accuracy: 0.9941451990632318
# Baseline malicious accuracy: 0.0 
# ----
# Adversarial regign accuracy: 0.28688524590163933
# Adversarial malicious accuracy: 0.0
# Adversarial malicious target accuracy: 1.0



# class Enforce255(art.defences.postprocessor.Postprocessor):
	# def __init__(self):
		# super().__init__()
	
	# def __call__(self, xs):
		# xs_slice = xs[:, 2:]
		# xs_slice = np.maximum(np.minimum(xs_slice * 255.0, 255.0), 0.0).astype(np.int32)
		# xs_slice = xs_slice.astype(np.float32) / 255.0
		# xs[:, 2:] = xs_slice
		# return xs



# print(ben_x)
# print(ben_y)
# print(mal_x)
# print(mal_y)
# print(mal_yt)
# print()
# print(ben_yh)
# print(mal_yh)
# print()
# print(ben_adv_x)
# print(ben_adv_yh)
# print()
# print(mal_adv_x)
# print(mal_adv_yh)



# plt.imshow(cosim_ben_adv)
# plt.title('Cosine similarities')
# plt.xlabel('ben_x')
# plt.ylabel('ben_adv_x')
# plt.colorbar()
# plt.show()
# plt.clf()

# plt.imshow(cosim_mal_adv)
# plt.title('Cosine similarities')
# plt.xlabel('mal_x')
# plt.ylabel('mal_adv_x')
# plt.colorbar()
# plt.show()
# plt.clf()

# plt.imshow(cosim_mal_advt)
# plt.title('Cosine similarities')
# plt.xlabel('mal_adv_x')
# plt.ylabel('ben_x')
# plt.colorbar()
# plt.show()
# plt.clf()
