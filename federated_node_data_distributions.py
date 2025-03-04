from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from library.random import split_key
from library.data import get_car_hacking_dataset


### setup

# RNG seeds
K0 = 999
K0, K1 = split_key(K0) # shuffle data

# hyperparameters
NODE_MIN = 1
NODE_MAX = 9
NODE_STEP = 1 # {1, 2, ..., 8}


### main

# load data
(train_x, train_y), _, _ = get_car_hacking_dataset(K1)

# undersample train set
###! train_x must be re-shuffled prior to node partitioning because RUS returns values sorted by class
_, train_y = RandomUnderSampler(random_state=K1).fit_resample(train_x, train_y)
train_indices = np.random.RandomState(seed=K1).permutation(len(train_y))
train_y = train_y[train_indices]

# split data
node_train_y_partitions = [np.array_split(train_y, n_nodes) for n_nodes in range(NODE_MIN, NODE_MAX, NODE_STEP)]

# determine distributions
dists = [[{int(k):v for k,v in Counter(y).items()} for y in typ] for typ in node_train_y_partitions]
print(dists)


### plotting

def plot_bar_charts(data):
	
	# init
	num_charts = len(data)
	classes = ['normal', 'dos', 'fuzzy', 'gear', 'rpm']
	fig, axes = plt.subplots(1, num_charts, figsize=(24, 4))
	
	for i, chart_data in enumerate(data):
		
		# prepare data
		x = np.arange(len(classes))
		y = [list(sublist.values()) for sublist in chart_data]
		y_norm = []
		for val in y:
			total = sum(val)
			normalized = [v / total for v in val]
			y_norm.append(normalized)
		
		# plot bars
		width = 0.15
		for j, val in enumerate(y_norm):
			axes[i].bar(x + j * width, val, width, label=f'Node {j + 1}')
		
		# draw title
		axes[i].set_title(f'{i + 1} Node'+('s' if i==0 else ''))
		
		# draw x axis
		axes[i].set_xticks(x + width * (len(y_norm) - 1) / 2, classes)
		axes[i].tick_params(axis='x', labelrotation=90)
		axes[i].set_xlabel('Class')
		
		# draw y axis
		axes[i].set_ylim(0, 1)
		if (i==0):
			axes[i].set_ylabel('Probability')
		else:
			axes[i].set_yticks([],[])
		
		# draw legend
		if i==(num_charts-1): axes[i].legend(loc='upper right')
	
	# finalise
	plt.tight_layout()
	plt.savefig('federated_node_data_distributions.png')

# call plot
plot_bar_charts(dists)
