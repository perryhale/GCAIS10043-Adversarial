import sys
import matplotlib.pyplot as plt
import numpy as np
import pickle

METRIC = sys.argv[1] if len(sys.argv)>1 else 'accuracy'

def load_history(filename):
	with open(filename, 'rb') as f:
		return pickle.load(f)

def extract_data(history):
	test_metric = np.array([[entry['test'][METRIC] for entry in row] for row in history])
	test_pgd_metric = np.array([[np.mean(entry['pgd'][METRIC]) for entry in row] for row in history])
	test_spn_metric = np.array([[np.mean(entry['spn'][METRIC]) for entry in row] for row in history])
	
	test_pgd_metric_drop = test_metric - test_pgd_metric
	test_spn_metric_drop = test_metric - test_spn_metric
	
	models_nc = np.array([[np.mean(entry['info']['n_components']) for entry in row] for row in history])
	models_hd = np.array([[np.mean(entry['info']['hidden_depth']) for entry in row] for row in history])
	
	return test_metric, test_pgd_metric_drop, test_spn_metric_drop, models_nc, models_hd

def plot_pixel_annotations(ax, image, txt_col='black', txt_size=6, txt_box=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1')):
	for i in range(image.shape[0]):
		for j in range(image.shape[1]):
			ax.text(j, i, f'{image[i,j]:.2f}', ha='center', va='center', color=txt_col, fontsize=txt_size, bbox=txt_box)

def plot_heatmaps(test_metric, test_pgd_metric_drop, test_spn_metric_drop):
	fig, axis = plt.subplots(nrows=1, ncols=3, figsize=(9, 5))
	
	axis[0].imshow(test_metric, cmap='Greens')
	axis[1].imshow(test_pgd_metric_drop, cmap='Reds')
	axis[2].imshow(test_spn_metric_drop, cmap='Reds')
	
	plot_pixel_annotations(axis[0], test_metric)
	plot_pixel_annotations(axis[1], test_pgd_metric_drop)
	plot_pixel_annotations(axis[2], test_spn_metric_drop)
	
	axis[0].set_title('Baseline accuracy')
	axis[1].set_title('Mean PGD accuracy drop')
	axis[2].set_title('Mean SPN accuracy drop')

	for ax in axis:
		ax.set_xlabel('Hidden depth')
		ax.set_ylabel('Principal components')
		ax.set_xticks(range(5))
		ax.set_yticks(range(10), range(1,11))

	fig.tight_layout()
	plt.savefig(f'data_pca_depth_train_{METRIC}_heatmaps.png')

def plot_scatter(axis, x, y, xlabel, ylabel, xlim, text_x):
	coef = np.polyfit(x, y, 1)
	pred = np.poly1d(coef)(x)
	rmse = np.sqrt(np.mean((y - pred) ** 2))
	corr = (coef[0] * np.var(x)) / (np.std(x, ddof=1) * np.std(y, ddof=1))#*(np.std(x, ddof=1) / np.std(y, ddof=1))
	
	axis.scatter(x, y, marker='+', c='red')
	axis.plot(np.sort(x), np.poly1d(coef)(np.sort(x)), c='black', linestyle='dashed')
	axis.set_xlabel(xlabel)
	axis.set_ylabel(ylabel)
	axis.grid()
	axis.set_ylim(0, 1)
	axis.set_xlim(*xlim)
	axis.text(text_x, 0.95, f'RMSE={rmse:.3f}')
	axis.text(text_x, 0.90, f'Correlation={corr:.3f}')

def plot_all_scatters(test_metric, test_pgd_metric_drop, test_spn_metric_drop, models_nc, models_hd):
	
	lim_pad = 0.25
	
	test_metric_flat = test_metric.reshape([-1])
	test_pgd_metric_drop_flat = test_pgd_metric_drop.reshape([-1])
	test_spn_metric_drop_flat = test_spn_metric_drop.reshape([-1])
	models_nc_flat = models_nc.reshape([-1])
	models_hd_flat = models_hd.reshape([-1])
	
	fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
	plot_scatter(axis[0], test_metric_flat, test_pgd_metric_drop_flat, 'Baseline accuracy', 'Mean PGD accuracy drop', (0, 1), 0.025)
	plot_scatter(axis[1], test_metric_flat, test_spn_metric_drop_flat, 'Baseline accuracy', 'Mean SPN accuracy drop', (0, 1), 0.025)
	plt.subplots_adjust(wspace=0.25)
	plt.savefig(f'data_pca_depth_train_scatter_base_{METRIC}_drop.png')
	
	fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
	plot_scatter(axis[0], models_nc_flat, test_pgd_metric_drop_flat, 'Principal components', 'Mean PGD accuracy drop', (1-lim_pad, 10+lim_pad), 1.025)
	plot_scatter(axis[1], models_nc_flat, test_spn_metric_drop_flat, 'Principal components', 'Mean SPN accuracy drop', (1-lim_pad, 10+lim_pad), 1.025)
	for ax in axis:
		ax.set_xticks(range(1, np.unique(models_nc_flat).shape[0]+1))
	plt.subplots_adjust(wspace=0.25)
	plt.savefig(f'data_pca_depth_train_scatter_ncomp_{METRIC}_drop.png')
	
	fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
	plot_scatter(axis[0], models_hd_flat, test_pgd_metric_drop_flat, 'Hidden depth', 'Mean PGD accuracy drop', (0-lim_pad, 4+lim_pad), 0.025)
	plot_scatter(axis[1], models_hd_flat, test_spn_metric_drop_flat, 'Hidden depth', 'Mean SPN accuracy drop', (0-lim_pad, 4+lim_pad), 0.025)
	for ax in axis:
		ax.set_xticks(range(np.unique(models_hd_flat).shape[0]))
	plt.subplots_adjust(wspace=0.25)
	plt.savefig(f'data_pca_depth_train_scatter_hdepth_{METRIC}_drop.png')

def main():
	history = load_history('data_pca_depth_train_history.pkl')
	test_metric, test_pgd_metric_drop, test_spn_metric_drop, models_nc, models_hd = extract_data(history)
	plot_heatmaps(test_metric, test_pgd_metric_drop, test_spn_metric_drop)
	plot_all_scatters(test_metric, test_pgd_metric_drop, test_spn_metric_drop, models_nc, models_hd)

if __name__ == "__main__":
	main()
