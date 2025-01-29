import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

# choose postfix
options = {'-u':'_uniform', '-s':'_scheduled'}
postfix = options[sys.argv[1]] if (len(sys.argv) > 1 and sys.argv[1] in options.keys()) else '_uniform'

# load dictionary
with open(f'bids_advpgd{postfix}_history_hevs.pkl', 'rb') as f:
	history = pickle.load(f)

# plot eigenspectra
def plot_eigenspectra(z=1):
	
	hevs = [list(reversed(abs(history[k]['test']['hevs'])))[::32] for k in history.keys()]
	min_y = min([min(h) for h in hevs])
	max_y = max([max(h) for h in hevs])
	
	fig, axis = plt.subplots(nrows=1, ncols=len(hevs), figsize=(21, 3))
	for i, (h, ax) in enumerate(zip(hevs, axis)):
		ax.bar(range(len(h)), h)
		ax.axhline(z, c='r', linestyle='dashed')
		ax.set_xticklabels([])
		ax.set_ylim(min_y, max_y)
		ax.set_yscale('log')
		ax.grid()
		if i!=0:
			ax.set_yticklabels([])
		else:
			ax.set_ylabel('log(abs(Eigenvalue))')
			ax.text(len(h)-5, z+0.7, f"z={z}", c='r', fontsize=10)
	
	plt.subplots_adjust(left=0.04, right=0.96, top=0.9, bottom=0.2, wspace=0.3, hspace=0.3)
	plt.savefig(f'bids_advpgd{postfix}_eigenspectra.png')


# plot evaluation metrics
def plot_evaluation(z=1):
	
	ms_axis = np.array([history[k]['max_strength'] for k in history.keys()])
	base_axis = np.array([history[k]['test']['accuracy'] for k in history.keys()])
	pgd_axis = np.array([np.mean(history[k]['pgd']['accuracy']) for k in history.keys()])
	spn_axis = np.array([np.mean(history[k]['spn']['accuracy']) for k in history.keys()])
	
	fig, axis = plt.subplots(nrows=1, ncols=7, figsize=(21, 3))
	ax1, ax2, ax3, ax4, ax5, ax6, ax7 = axis
	
	ax1.plot(ms_axis, base_axis, c='g')
	ax2.plot(ms_axis, base_axis - pgd_axis, c='r')
	ax3.plot(ms_axis, base_axis - spn_axis, c='r')
	ax4.plot(ms_axis, [np.sum([e/(e+z) for e in h]) for h in [history[k]['test']['hevs'] for k in history.keys()]], c='lightsteelblue')
	ax5.plot(ms_axis, [np.sum([e/(e+z) for e in h]) for h in [list(reversed(sorted(abs(history[k]['test']['hevs'])))) for k in history.keys()]], c='lightsteelblue')
	ax6.plot(ms_axis, [np.sum([e/(e+z) for e in h]) for h in [list(reversed(sorted(abs(history[k]['test']['hevs']))))[:512] for k in history.keys()]], c='lightsteelblue')
	ax7.plot(ms_axis, [np.sum([e/(e+z) for e in h]) for h in [list(reversed(history[k]['test']['hevs']))[:512] for k in history.keys()]], c='lightsteelblue')
	
	ax1.set_ylabel('Baseline accuracy')
	ax2.set_ylabel('Mean PGD accuracy drop')
	ax3.set_ylabel('Mean SPN accuracy drop')
	ax4.set_ylabel(f'Neff-complete (z={z})')
	ax5.set_ylabel(f'Neff-abs-complete (z={z})')
	ax6.set_ylabel(f'Neff-abs-512 (z={z})')
	ax7.set_ylabel(f'Neff-512 (z={z})')
	
	for ax in axis:
		ax.set_xlabel('max_strength')
		ax.grid()
	
	plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.22, wspace=0.4, hspace=0.3)
	plt.savefig(f'bids_advpgd{postfix}_evaluation_hevs.png')


# call plot functions
z = 1
plot_eigenspectra(z)
plot_evaluation(z)

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
