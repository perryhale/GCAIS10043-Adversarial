import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

from imblearn.under_sampling import RandomUnderSampler
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import DecisionBoundaryDisplay

from library.random import split_key
from library.data import get_car_hacking_dataset


### setup

# timer
T0 = time.time()
print(f'[Elapsed time: {time.time()-T0:.2f}s]')

# RNG seed
K0 = 999
K0, K1 = split_key(K0) # shuffle data


### functions

def decision_plot(clf, x, y, cmap=None, s=8, edgecolors='k', linewidth=0.2):
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.set_xlim(min(x[:, 0]), max(x[:, 0]))
	ax.set_ylim(min(x[:, 1]), max(x[:, 1]))
	#ax.set_xlim(0, 1)
	#ax.set_ylim(0, 1)
	disp = DecisionBoundaryDisplay.from_estimator(
		clf,
		x[:, :FEATURES_DIM],
		response_method='predict',
		cmap=cmap,
		ax=ax,
		xlabel='PC1',
		ylabel='PC2',
		grid_resolution=300
	)
	ax.scatter(
		x[:, 0], 
		x[:, 1], 
		c=y, 
		cmap=cmap, 
		s=s, 
		edgecolors=edgecolors, 
		linewidth=linewidth
	)
	ax.set_title(f'{kernel} SVM decision boundaries')
	return fig, ax


### hyperparameters

# data
FEATURES_RES = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255]).astype('float32')
LABELS_NAMES = ['Normal', 'DOS', 'Fuzzy', 'Gear', 'RPM']
LABELS_DIM = 5
FEATURES_DIM = 2

# plotting
KERNEL_SPACE = ['linear', 'rbf', 'poly', 'sigmoid']
CMAP = plt.cm.Paired

# tracing
VERBOSE = False


### prepare data

# load partitions
(train_x, train_y), (val_x, val_y), (test_x, test_y) = get_car_hacking_dataset(K1)

# undersample train set
train_x, train_y = RandomUnderSampler(random_state=K1).fit_resample(train_x, train_y)

# standardize features
train_x_std = scale(train_x)
val_x_std = scale(val_x)
test_x_std = scale(test_x)

# fit data transform objects
data_pca = PCA().fit(train_x_std)
train_x_pct = data_pca.transform(train_x_std)
data_mms = MinMaxScaler().fit(train_x_pct)

# transform and normalise data
train_x_pctmms = data_mms.transform(train_x_pct)
val_x_pctmms = data_mms.transform(data_pca.transform(val_x_std))
test_x_pctmms = data_mms.transform(data_pca.transform(test_x_std))

# trace
print(train_x.shape, train_y.shape, 'train')
print(val_x.shape, val_y.shape, 'val')
print(test_x.shape, test_y.shape, 'test')
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### fit classifers

# init gridsearch
history = {}

# run gridsearch
for kernel in KERNEL_SPACE:
	
	# fit svm
	clf = svm.SVC(
		C=1.0,
		kernel=kernel,
		degree=4,
		decision_function_shape='ovo',
		verbose=VERBOSE)
	clf.fit(train_x_pctmms[:, :FEATURES_DIM], train_y)
	
	# evaluate on test set
	test_yh = clf.predict(test_x_pctmms[:, :FEATURES_DIM])
	test_accuracy = accuracy_score(test_yh, test_y)
	test_recall = recall_score(test_yh, test_y, average='weighted', zero_division=0.0)
	test_precision = precision_score(test_yh, test_y, average='weighted')
	test_f1score = f1_score(test_yh, test_y, average='weighted')
	history.update({kernel:{
		'accuracy':test_accuracy,
		'recall':test_recall,
		'precision':test_precision,
		'f1score':test_f1score
	}})
	
	# trace
	print(kernel)
	print(history[kernel])
	print(f'[Elapsed time: {time.time()-T0:.2f}s]')
	
	# plot decision boundary and test set
	fig, ax = decision_plot(clf, test_x_pctmms[:, :FEATURES_DIM], test_y, cmap=CMAP)
	plt.savefig(f'data_pca_svm_{kernel}_test.png')
	print(f'[Elapsed time: {time.time()-T0:.2f}s]')
	
	# plot decision boundary and train set
	fig, ax = decision_plot(clf, train_x_pctmms[:, :FEATURES_DIM], train_y, cmap=CMAP)
	plt.savefig(f'data_pca_svm_{kernel}_train.png')
	print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### plot bar chart

all_metrics = history[KERNEL_SPACE[0]].keys()
all_bar_heights = [[history[k][metric] for k in KERNEL_SPACE] for metric in all_metrics]
bar_colors = ['blue', 'lightsteelblue']
bar_index = range(len(KERNEL_SPACE))

fig, axis = plt.subplots(nrows=1, ncols=len(all_metrics), figsize=(18,5))

for metric, bar_heights, ax in zip (all_metrics, all_bar_heights, axis):
	ax.bar(bar_index, bar_heights, color=bar_colors)
	ax.set_xticks(bar_index, KERNEL_SPACE)
	ax.set_ylabel(f'Test {metric}')
	for index, height in enumerate(bar_heights):
		ax.text(index-0.38, height+0.01, f'{height:.3f}', c='black', fontsize='medium')
	plt.savefig('data_pca_svm_test_accuracy.png')
	print(f'[Elapsed time: {time.time()-T0:.2f}s]')
