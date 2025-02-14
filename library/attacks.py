from abc import ABC
import pandas as pd
import numpy as np
from tqdm import tqdm
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import ProjectedGradientDescent
from .data import enforce_res
from .models import classifier_evaluation

"""

Future refactor notes:
*classifier_evaluation functions could be combined into a function that takes an attack factory lambda
see below, just need to consolidate interface

"""


# define interface
class AbstractAttack(ABC):
	
	# type: (AbstractAttack, np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
	def generate(self, x, y, mask=None):
		return x


# SPN attack
class SaltAndPepperNoise(AbstractAttack):
	def __init__(self, noise_ratio=0.1, noise_magnitude=1.0):
		self.noise_ratio = noise_ratio
		self.noise_magnitude = noise_magnitude
	
	def generate(self, x, y, mask=None):
		###! redundant arg y, included for interface consistency
		noise = np.random.uniform(-1, 1, x.shape).astype(np.float32)
		noise = np.where(np.abs(noise) <= self.noise_ratio, self.noise_magnitude * (np.sign(noise) / 2 + 0.5) - x, 0)
		if mask is not None:
			noise *= mask
		return x + noise

# x = np.ones((32,32,1)) / 2
# plt.imshow(x)
# plt.colorbar()
# plt.show()
# xh = SaltAndPepperNoise().generate(x)
# plt.imshow(xh)
# plt.colorbar()
# plt.show()
# import sys;sys.exit()


# Benign-any malicious / any malicious-benign PGD attack
class BenMalPGD(AbstractAttack):
	def __init__(self, model, input_dim, output_dim, criterion, epsilon=0.1, iterations=7, clip_values=(0,1), batch_size=32, verbose=False):
		art_model = TensorFlowV2Classifier(
			model=model,
			input_shape=(input_dim,),
			nb_classes=output_dim,
			loss_object=criterion,
			clip_values=clip_values
		)
		self.pgd_untargeted = ProjectedGradientDescent(
			estimator=art_model,
			eps=epsilon,
			eps_step=epsilon/iterations,
			max_iter=iterations,
			num_random_init=1,
			targeted=False,
			batch_size=batch_size,
			verbose=verbose
		)
		self.pgd_targeted = ProjectedGradientDescent(
			estimator=art_model,
			eps=epsilon,
			eps_step=epsilon/iterations,
			max_iter=iterations,
			num_random_init=1,
			targeted=True,
			batch_size=batch_size,
			verbose=verbose
		)
	
	def generate(self, x, y, mask=None):
		
		# seperate classes
		ben_x = x[y==0]
		ben_y = y[y==0]
		ben_mask = mask[y==0] if mask is not None else None
		mal_x = x[y>=1]
		mal_y = y[y>=1]
		mal_yt = np.zeros_like(mal_y)
		mal_mask = mask[y>=1] if mask is not None else None
		
		# generate samples
		ben_adv_x = self.pgd_untargeted.generate(ben_x, mask=ben_mask)
		mal_adv_x = self.pgd_targeted.generate(mal_x, mal_yt, mask=mal_mask)
		
		return np.concatenate([ben_adv_x, mal_adv_x])


"""
type: (
	.
) ->
"""
def benmalpgd_classifier_evaluation(
		model,
		input_dim,
		output_dim,
		loss_fn,
		x,
		y,
		mask=None,
		feature_res=None,
		eps_min=0.0,
		eps_max=1.0,
		eps_num=8,
		pgd_iter=7,
		batch_size=64,
		score_average='weighted',
		verbose=False
	):
	
	history = {
		'epsilon':[],
		'loss':[],
		'accuracy':[],
		'precision':[],
		'recall':[],
		'fscore':[],
		'confusion':[]
	}
	for epsilon in tqdm(np.linspace(eps_min, eps_max, num=eps_num), desc='BMPGD Eval', unit='epsilon'):
		
		# generate samples
		attack = BenMalPGD(model, input_dim, output_dim, loss_fn, epsilon=epsilon, iterations=pgd_iter, batch_size=batch_size, verbose=verbose)
		adv_x = attack.generate(x, y, mask=mask)
		if feature_res is not None:
			adv_x = enforce_res(adv_x, feature_res)
		
		# evaluate model
		test_history = classifier_evaluation(model, loss_fn, adv_x, y, batch_size=batch_size, score_average=score_average)
		
		# update attack history
		history['epsilon'].append(epsilon.item())
		for hkey in test_history.keys():
			history[hkey].append(test_history[hkey])
	
	return history


"""
type: (
	.
) ->
"""
def spn_classifier_evaluation(
		model,
		loss_fn,
		x,
		y,
		mask=None,
		feature_res=None,
		eps_min=0.0,
		eps_max=1.0,
		eps_num=8,
		spn_magnitude=1.0,
		batch_size=64,
		score_average='weighted',
		verbose=False
	):
	
	history = {
		'epsilon':[],
		'loss':[],
		'accuracy':[],
		'precision':[],
		'recall':[],
		'fscore':[],
		'confusion':[]
	}
	for epsilon in tqdm(np.linspace(eps_min, eps_max, num=eps_num), desc='SPN Eval', unit='epsilon'):
		
		# generate samples
		attack = SaltAndPepperNoise(noise_ratio=epsilon, noise_magnitude=spn_magnitude)
		adv_x = attack.generate(x, y, mask=mask)
		if feature_res is not None:
			adv_x = enforce_res(adv_x, feature_res)
		
		# evaluate model
		test_history = classifier_evaluation(model, loss_fn, adv_x, y, batch_size=batch_size, score_average=score_average)
		
		# update attack history
		history['epsilon'].append(epsilon.item())
		for hkey in test_history.keys():
			history[hkey].append(test_history[hkey])
	
	return history


###! experimental
"""
type: (
	Dict[str : lambda (float) -> AbstractAttack],
	tf.keras.losses.Loss,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	np.ndarray,
	float,
	float,
	int,
	int,
	bool
) -> Dict[str : Dict[str : List[float]]]
"""
"""
attack_init_dict = {
	'bmpgd' : lambda epsilon : BenMalPGD(
		model,
		FEATURES_DIM,
		LABELS_DIM,
		criterion,
		epsilon=epsilon,
		iterations=PGD_ITER,
		batch_size=BATCH_SIZE,
		verbose=VERBOSE
	),
	'spn' : lambda epsilon : SaltAndPepperNoise(
		noise_ratio=epsilon,
		noise_magnitude=SPN_MAG
	)
}
"""
# def adversarial_classifier_evaluation(
		# model,
		# attack_init_dict,
		# loss_fn,
		# x,
		# y,
		# mask=None,
		# feature_res=None,
		# eps_min=0.0,
		# eps_max=1.0,
		# eps_num=8,
		# batch_size=64,
		# score_average='weighted',
		# verbose=False
	# ):
	
	# # init history
	# history = {key:{} for key in attack_init_dict.keys()}
	
	# # populate history
	# for key in attack_init_dict.keys():
		
		# # init attack history
		# attack_history = {
			# 'epsilon':[],
			# 'loss':[],
			# 'accuracy':[],
			# 'confusion':[],
			# 'precision':[],
			# 'recall':[],
			# 'fscore':[]
		# }
		
		# # run gridsearch
		# for epsilon in tqdm(np.linspace(eps_min, eps_max, num=eps_num), desc=f'{key.upper()} eval', unit='epsilon'):
			
			# # generate samples
			# attack = attack_init_dict[key](epsilon)
			# adv_x = attack.generate(x, y, mask=mask)
			# if feature_res is not None:
				# adv_x = enforce_res(adv_x, feature_res)
			
			# # evaluate model
			# test_history = classifier_evaluation(model, loss_fn, adv_x, y, batch_size=batch_size, score_average=score_average)
			
			# # update attack history
			# attack_history['epsilon'].append(epsilon.item())
			# for hkey in test_history.keys():
				# attack_history[hkey].append(test_history[hkey])
		
		# # update history
		# history.update({key:attack_history})
	
	# return history
