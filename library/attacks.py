from abc import ABC
import pandas as pd
import numpy as np
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import ProjectedGradientDescent


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
	def __init__(self, model, input_dim, output_dim, criterion, epsilon=0.1, iterations=7, batch_size=32, verbose=False):
		art_model = TensorFlowV2Classifier(
			model=model,
			input_shape=(input_dim,),
			nb_classes=output_dim,
			loss_object=criterion,
			clip_values=(0,1)
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
		
		# merge [x, y, mask] and split by class category
		data = pd.DataFrame(np.concatenate([x, y.reshape((y.shape[0], 1,)), mask], axis=-1))
		data_ben = data.loc[data[10] == 0]
		data_mal = data.loc[data[10] >= 1]
		
		# unpack dataframe
		ben_x = data_ben.iloc[:, :10].to_numpy()
		ben_y = data_ben.iloc[:, 10].to_numpy()
		ben_mask = data_ben.iloc[:, 11:].to_numpy()
		mal_x = data_mal.iloc[:, :10].to_numpy()
		mal_y = data_mal.iloc[:, 10].to_numpy()
		mal_mask = data_mal.iloc[:, 11:].to_numpy()
		mal_yt = np.zeros(len(mal_y))
		
		# generate samples
		ben_adv_x = self.pgd_untargeted.generate(ben_x, mask=ben_mask)
		mal_adv_x = self.pgd_targeted.generate(mal_x, mal_yt, mask=mal_mask)
		
		return np.concatenate([ben_adv_x, mal_adv_x])
