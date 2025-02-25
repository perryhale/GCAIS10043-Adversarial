import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle

METRIC = sys.argv[1] if len(sys.argv) > 1 else 'accuracy'

def load_history(filename):
	with open(filename, 'rb') as f:
		return pickle.load(f)

def extract_data(history):
	test_metric = np.array([[entry['test'][METRIC] for entry in row] for row in history])
	test_pgd_metric = np.array([[np.mean(entry['pgd'][METRIC]) for entry in row] for row in history])
	test_spn_metric = np.array([[np.mean(entry['spn'][METRIC]) for entry in row] for row in history])
	
	test_pgd_metric_drop = test_metric - test_pgd_metric
	test_spn_metric_drop = test_metric - test_spn_metric
	
	models_nn = np.array([[np.mean(entry['info']['n_nodes']) for entry in row] for row in history])
	models_ms = np.array([[np.mean(entry['info']['max_strength']) for entry in row] for row in history])
	
	return test_metric, test_pgd_metric_drop, test_spn_metric_drop, models_nn, models_ms

def plot_pixel_annotations(ax, image, txt_col='black', txt_size=6, txt_box=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1')):
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			ax.text(j, i, f'{image[i, j]:.2f}', ha='center', va='center', color=txt_col, fontsize=txt_size, bbox=txt_box)

def plot_heatmaps(test_metric, test_pgd_metric_drop, test_spn_metric_drop):
	fig, axis = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
	
	for ax, data, title in zip(
			axis, 
			[test_metric, test_pgd_metric_drop, test_spn_metric_drop],
			[f'Baseline {METRIC.replace("fscore","f1-score")}', f'Mean PGD {METRIC.replace("fscore","f1-score")} drop', f'Mean SPN {METRIC.replace("fscore","f1-score")} drop']
		):
		ax.imshow(data, cmap='Reds' if 'drop' in title else 'Greens')
		plot_pixel_annotations(ax, data)
		ax.set_title(title)
		ax.set_xlabel('n_nodes')
		ax.set_ylabel('max_strength')
		ax.set_xticks(range(data.shape[1]), range(1, data.shape[0]+1))
		ax.set_yticks(range(data.shape[0]), [f'{n:.2f}' for n in np.linspace(0, 0.5, num=8)])
	
	fig.tight_layout()
	plt.savefig(f'federated_adversarial_train_pgd_uniform_{METRIC.replace("fscore","f1-score")}_heatmaps.png')

def plot_scatter(axis, x, y, xlabel, ylabel):
	coef = np.polyfit(x, y, 1)
	pred = np.poly1d(coef)(x)
	rmse = np.sqrt(np.mean((y - pred) ** 2))
	corr = np.corrcoef(x, y)[0, 1]
	
	axis.scatter(x, y, marker='+', c='red')
	axis.plot(np.sort(x), np.poly1d(coef)(np.sort(x)), c='black', linestyle='dashed')
	axis.set_xlabel(xlabel)
	axis.set_ylabel(ylabel)
	axis.grid()
	
	#axis.set_ylim(min(y) - 0.05*(max(y)-min(y)), max(y) + 0.05*(max(y)-min(y)))
	#axis.set_xlim(min(x) - 0.05*(max(x)-min(x)), max(x) + 0.05*(max(x)-min(x)))
	#axis.text(min(x) + 0.02, max(y) - 0.05, f'RMSE={rmse:.3f}\nCorrelation={corr:.3f}')
	axis.text(
		axis.get_xlim()[0] + 0.02 * (axis.get_xlim()[1] - axis.get_xlim()[0]), 
		axis.get_ylim()[1] - 0.05 * (axis.get_ylim()[1] - axis.get_ylim()[0]),
		f'RMSE={rmse:.3f}\nCorrelation={corr:.3f}',
		verticalalignment='top', horizontalalignment='left'
	)

def plot_all_scatters(test_metric, test_pgd_metric_drop, test_spn_metric_drop, models_nn, models_ms):
	test_metric_flat = test_metric.flatten()
	test_pgd_metric_drop_flat = test_pgd_metric_drop.flatten()
	test_spn_metric_drop_flat = test_spn_metric_drop.flatten()
	models_nn_flat = models_nn.flatten()
	models_ms_flat = models_ms.flatten()
	
	fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
	for ax in axis:
		ax.set_xlim(0,1)
		ax.set_ylim(0,1)
	plot_scatter(axis[0], test_metric_flat, test_pgd_metric_drop_flat, f'Baseline {METRIC.replace("fscore","f1-score")}', f'Mean PGD {METRIC.replace("fscore","f1-score")} drop')
	plot_scatter(axis[1], test_metric_flat, test_spn_metric_drop_flat, f'Baseline {METRIC.replace("fscore","f1-score")}', f'Mean SPN {METRIC.replace("fscore","f1-score")} drop')
	plt.subplots_adjust(wspace=0.25)
	plt.savefig(f'federated_adversarial_train_pgd_uniform_scatter_base_{METRIC.replace("fscore","f1-score")}_drop.png')
	
	fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
	for ax in axis:
		ax.set_xticks(range(1,9))
		ax.set_ylim(0,1)
	plot_scatter(axis[0], models_nn_flat, test_pgd_metric_drop_flat, 'Num federated nodes', f'Mean PGD {METRIC.replace("fscore","f1-score")} drop')
	plot_scatter(axis[1], models_nn_flat, test_spn_metric_drop_flat, 'Num federated nodes', f'Mean SPN {METRIC.replace("fscore","f1-score")} drop')
	plt.subplots_adjust(wspace=0.25)
	plt.savefig(f'federated_adversarial_train_pgd_uniform_scatter_nnodes_{METRIC.replace("fscore","f1-score")}_drop.png')
	
	fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
	for ax in axis:
		ax.set_xticks(np.linspace(0,0.5,num=8), [f'{n:.2f}' for n in np.linspace(0,0.5,num=8)])
		ax.set_ylim(0,1)
	plot_scatter(axis[0], models_ms_flat, test_pgd_metric_drop_flat, 'Max strength', f'Mean PGD {METRIC.replace("fscore","f1-score")} drop')
	plot_scatter(axis[1], models_ms_flat, test_spn_metric_drop_flat, 'Max strength', f'Mean SPN {METRIC.replace("fscore","f1-score")} drop')
	plt.subplots_adjust(wspace=0.25)
	plt.savefig(f'federated_adversarial_train_pgd_uniform_scatter_mstrength_{METRIC.replace("fscore","f1-score")}_drop.png')

def main():
	history = load_history('federated_adversarial_train_pgd_uniform_history.pkl')
	test_metric, test_pgd_metric_drop, test_spn_metric_drop, models_nn, models_ms = extract_data(history)
	plot_heatmaps(test_metric, test_pgd_metric_drop, test_spn_metric_drop)
	plot_all_scatters(test_metric, test_pgd_metric_drop, test_spn_metric_drop, models_nn, models_ms)

if __name__ == "__main__":
	main()
