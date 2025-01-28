import pickle
import numpy as np
import matplotlib.pyplot as plt

# load dictionary
with open('bids_advpgd_uniform_history.pkl', 'rb') as f:
	history = pickle.load(f)

# plot train curves
def plot_train():
	
	epoch_axis = [i+1 for i in range(len(history[list(history.keys())[0]]['train']['loss']))]
	
	fig, axis = plt.subplots(nrows=2, ncols=len(history.keys()), figsize=(21, 5))
	row1, row2 = axis
	
	row1_miny = min([min([min(history[k]['train']['loss']), min(history[k]['train']['val_loss']), history[k]['test']['loss']]) for k in history.keys()])
	row1_maxy = max([max([max(history[k]['train']['loss']), max(history[k]['train']['val_loss']), history[k]['test']['loss']]) for k in history.keys()])
	
	for i, (ax, k) in enumerate(zip(row1, history.keys())):
		ax.set_title(k, fontsize=8)
		ax.plot(epoch_axis, history[k]['train']['loss'], label='train')
		ax.plot(epoch_axis, history[k]['train']['val_loss'], label='validation', c='r')
		ax.scatter([len(epoch_axis)], [history[k]['test']['loss']], marker='x', label='test', c='g')
		ax.set_xticks(epoch_axis, [])
		ax.set_yticks(np.linspace(row1_miny, row1_maxy, num=5))
		ax.set_ylim(row1_miny, row1_maxy)
		ax.grid()
		if i==0:
			ax.legend(prop={'size': 8})
			ax.set_ylabel('Loss')
		else:
			ax.set_yticklabels([])
	
	for i, (ax, k) in enumerate(zip(row2, history.keys())):
		ax.plot(epoch_axis, history[k]['train']['accuracy'], label='train', c='blue')
		ax.plot(epoch_axis, history[k]['train']['val_accuracy'], label='validation', c='red')
		ax.plot(epoch_axis, history[k]['train']['adv_accuracy'], label='adv train', c='blue', linestyle='dashed')
		ax.plot(epoch_axis, history[k]['train']['val_adv_accuracy'], label='adv validation', c='red', linestyle='dashed')
		ax.scatter([len(epoch_axis)], [history[k]['test']['accuracy']], marker='x', label='test', c='g')
		ax.set_xlabel('Epoch')
		ax.set_xticks(epoch_axis, epoch_axis)
		ax.set_ylim(0, 1)
		ax.grid()
		if i==0:
			ax.legend(prop={'size': 8})
			ax.set_ylabel('Accuracy')
		else:
			ax.set_yticklabels([])
	
	plt.subplots_adjust(left=0.04, right=0.96, top=0.9, bottom=0.1, wspace=0.3)
	plt.savefig('bids_advpgd_uniform_train.png')

# plot adversarial evaluation
def plot_adversarial():
	
	pgd_axis = history[list(history.keys())[0]]['pgd']['epsilon']
	spn_axis = history[list(history.keys())[0]]['spn']['ratio']
	
	fig, axis = plt.subplots(nrows=2, ncols=len(history.keys()), figsize=(21, 5))
	row1, row2 = axis
	
	for i, (ax, k) in enumerate(zip(row1, history.keys())):
		ax.set_title(k, fontsize=8)
		ax.plot(pgd_axis, history[k]['pgd']['accuracy'], c='r')
		ax.set_xlabel('ε')
		ax.set_ylim(0, 1)
		ax.setyticks(pgd_axis, pgd_axis[::2])
		ax.grid()
		if i==0:
			ax.set_ylabel('PGD test accuracy')
		else:
			ax.set_yticklabels([])
	
	for i, (ax, k) in enumerate(zip(row2, history.keys())):
		ax.plot(spn_axis, history[k]['spn']['accuracy'], c='r')
		ax.set_xlabel('noise_ratio')
		ax.set_ylim(0, 1)
		ax.grid()
		if i==0:
			ax.set_ylabel('SPN test accuracy')
		else:
			ax.set_yticklabels([])
	
	plt.subplots_adjust(left=0.04, right=0.96, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
	plt.savefig('bids_advpgd_uniform_adversarial.png')

# annotate image pixels util fn
def annotate_image(ax, image, txt_col='black', txt_size=8, txt_box=None):
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			ax.text(j, i, f'{image[i,j]:.2f}', ha='center', va='center', color=txt_col, fontsize=txt_size, bbox=txt_box)

# plot confusion
def plot_confusion():
	
	fig, axis = plt.subplots(nrows=1, ncols=len(history.keys()), figsize=(21, 3))
	
	for ax, k in zip(axis, history.keys()):
		cfm = history[k]['test']['confusion']
		ax.imshow(cfm)
		ax.set_xticks(range(5), range(5))
		ax.set_yticks(range(5), range(5))
		annotate_image(ax, cfm)
	
	plt.subplots_adjust(left=0.04, right=0.96, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
	plt.savefig('bids_advpgd_uniform_confusion.png')

# plot evaluation metrics
def plot_evaluation():
	
	ms_axis = [history[k]['max_strength'] for k in history.keys()]
	base_axis = [history[k]['test']['accuracy'] for k in history.keys()]
	pgd_axis = [np.sum(history[k]['pgd']['accuracy']) for k in history.keys()]
	spn_axis = [np.sum(history[k]['spn']['accuracy']) for k in history.keys()]
	
	fig, axis = plt.subplots(nrows=1, ncols=3, figsize=(21, 3))
	ax1, ax2, ax3 = axis
	
	ax1.plot(ms_axis, base_axis, c='g')
	ax1.plot(ms_axis, pgd_axis, c='r')
	ax1.plot(ms_axis, spn_axis, c='r')
	
	ax1.set_ylabel('Baseline accuracy')
	ax2.set_ylabel('Cumulative PGD accuracy')
	ax3.set_ylabel('Cumulative SPN accuracy')
	
	for ax in axis:
		ax.set_xlabel('max_strength')
		ax.grid()
	
	plt.savefig('bids_advpgd_uniform_evaluation.png')


# call plot functions
plot_train()
plot_adversarial()
plot_confusion()
plot_evaluation()

# print keys (after gtk error msg spam)
print(history)
print(list(history.keys()))
print(list(history[list(history.keys())[0]].keys()))
for k in history[list(history.keys())[0]].keys(): 
	elem = history[list(history.keys())[0]][k]
	if 'keys' in dir(elem):
		print(k, list(elem.keys()))
	else:
		print(k, elem)
