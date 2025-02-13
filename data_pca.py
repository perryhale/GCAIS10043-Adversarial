import os
import time

import numpy as np
#import matplotlib; matplotlib.use('Qt5Agg') # patch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from imblearn.under_sampling import RandomUnderSampler

from library.random import split_key
from library.data import get_car_hacking_dataset


### hyperparameters

# timer
T0 = time.time()
print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# RNG seeds
K0, K1, K2 = split_key(999, n=3) # data

# data
FEATURES_RES = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255]).astype('float32')
TRUNCATION = 50_000

# plotting
CLASS_NAMES = ['Normal', 'DOS', 'Fuzzy', 'Gear', 'RPM']
CLASS_COLS = ['black', 'red', 'green', 'blue', 'orange']
FEATURE_NAMES = ['ID', 'DLC', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
PLOT_MARKER = 'o'
PC_X = 0
PC_Y = 1


### prepare data

# load and normalise
data_x, data_y = zip(*get_car_hacking_dataset(K1))
data_x = scale(np.concatenate(data_x))
data_y = np.concatenate(data_y)

# truncate
data_x = data_x[:TRUNCATION]
data_y = data_y[:TRUNCATION]

# undersample
data_x, data_y = RandomUnderSampler(random_state=K2).fit_resample(data_x, data_y)

# transform onto principal axis
data_pca = PCA().fit(data_x)
data_x_pc10 = data_pca.transform(data_x)

# partition by class
data_x_classes = [data_x[data_y == i] for i in range(len(CLASS_NAMES))]
data_x_pc10_classes = [data_x_pc10[data_y == i] for i in range(len(CLASS_NAMES))]

# trace
for data_x, class_name in zip(data_x_pc10_classes, CLASS_NAMES):
	print(data_x.shape[0], class_name)
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### plotting

# 3d projection
def animate_3d_update(frame, ax, data_x_pc10_classes, CLASS_NAMES, CLASS_COLS):
	ax.clear()
	for data_x, class_name, class_col in zip(data_x_pc10_classes, CLASS_NAMES, CLASS_COLS):
		ax.scatter(*data_x[:, :3].T, s=8, c=class_col, marker=PLOT_MARKER, label=class_name)
	ax.set_xlabel('PC1')
	ax.set_ylabel('PC2')
	ax.set_zlabel('PC3')
	ax.view_init(elev=30, azim=frame)  # Adjust the angle here (frame is the current angle)
	ax.legend()
	return ax

def animate_3d():
	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(projection='3d')
	ani = FuncAnimation(
		fig,
		animate_3d_update,
		frames=np.arange(0, 360, 1),
		fargs=(ax, data_x_pc10_classes, CLASS_NAMES, CLASS_COLS),
		interval=50,
		repeat=True
	)
	ani.save('data_pca_3d.mp4', writer='ffmpeg', fps=30)

def interactive_3d():
	fig = plt.figure(figsize=(8, 8))
	ax = fig.add_subplot(projection='3d')
	for data_x, class_name, class_col in zip(data_x_pc10_classes, CLASS_NAMES, CLASS_COLS):
		ax.scatter(*data_x[:,:3].T, s=8, c=class_col, marker=PLOT_MARKER, label=class_name)

	ax.legend()
	ax.set_xlabel('PC1')
	ax.set_ylabel('PC2')
	ax.set_zlabel('PC3')
	fig.tight_layout()
	plt.show()

#interactive_3d()
animate_3d()

# overall scree
fig, ax = plt.subplots()
ax.bar(range(10), data_pca.explained_variance_ratio_)
ax.set_xlabel('Principle component')
ax.set_ylabel('Explained variance ratio')
plt.savefig('data_pca_scree_all.png')

# class screes
for data_x, class_name, class_col in zip(data_x_classes, CLASS_NAMES, CLASS_COLS):
	data_x_pca = PCA().fit(data_x)
	
	fig, ax = plt.subplots()
	fig.suptitle(f'{class_name} scree')
	ax.bar(range(10), data_x_pca.explained_variance_ratio_, color=class_col)
	ax.set_xlabel('Principle component')
	ax.set_ylabel('Explained variance ratio')
	plt.savefig(f'data_pca_scree_{class_name}.png'.lower())

# overall_loading scores
fig, axis = plt.subplots(nrows=2, ncols=5, figsize=(14, 6))
fig.suptitle(f'Overall loading scores')
row1, row2 = axis
for i, eigenvector, ax in zip(range(0,5), data_pca.components_[:5], row1):
	ax.set_title(f'PC{i}')
	ax.bar(range(len(eigenvector)), eigenvector)
for i, eigenvector, ax in zip(range(5,10), data_pca.components_[5:], row2):
	ax.set_title(f'PC{i}')
	ax.bar(range(len(eigenvector)), eigenvector)
for i, row in enumerate(axis):
	for j, ax in enumerate(row):
		ax.set_ylim(-1,1)
		if i==1:
			ax.set_xticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=90)
		else:
			ax.set_xticks(range(len(FEATURE_NAMES)), [])
		if j==0:
			ax.set_ylabel('Loading score')
		else:
			ax.set_yticks([])
fig.tight_layout()
plt.savefig('data_pca_load_all.png')

# class loading scores
for data_x, class_name, class_col in zip(data_x_classes, CLASS_NAMES, CLASS_COLS):
	data_x_pca = PCA().fit(data_x)
	
	fig, axis = plt.subplots(nrows=2, ncols=5, figsize=(14, 6))
	fig.suptitle(f'{class_name} loading scores')
	row1, row2 = axis
	for i, eigenvector, ax in zip(range(0,5), data_x_pca.components_[:5], row1):
		ax.set_title(f'PC{i}')
		ax.bar(range(len(eigenvector)), eigenvector, color=class_col)
	for i, eigenvector, ax in zip(range(5,10), data_x_pca.components_[5:], row2):
		ax.set_title(f'PC{i}')
		ax.bar(range(len(eigenvector)), eigenvector, color=class_col)
	for i, row in enumerate(axis):
		for j, ax in enumerate(row):
			ax.set_ylim(-1,1)
			if i==1:
				ax.set_xticks(range(len(FEATURE_NAMES)), FEATURE_NAMES, rotation=90)
			else:
				ax.set_xticks(range(len(FEATURE_NAMES)), [])
			if j==0:
				ax.set_ylabel('Loading score')
			else:
				ax.set_yticks([])
	fig.tight_layout()
	plt.savefig(f'data_pca_load_{class_name}.png'.lower())

# 2d projection
fig, ax = plt.subplots(figsize=(8, 8))
for data_x, class_name, class_col in zip(data_x_pc10_classes, CLASS_NAMES, CLASS_COLS):
	ax.scatter(data_x[:,PC_X], data_x[:,PC_Y], s=8, c=class_col, marker=PLOT_MARKER, label=class_name)

ax.legend()
ax.set_xlabel(f'PC{PC_X+1}')
ax.set_ylabel(f'PC{PC_Y+1}')
fig.tight_layout()
plt.savefig('data_pca_2d.png')
