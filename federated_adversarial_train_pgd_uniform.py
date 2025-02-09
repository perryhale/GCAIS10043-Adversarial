import os
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, optimizers, callbacks
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm import tqdm
import pickle

from library.random import split_key
from library.data import enforce_res, mask_fn, get_car_hacking_dataset
from library.models import get_multiclass_mlp
from library.training import federated_uniform_adversarial_train
from library.attacks import BenMalPGD, SaltAndPepperNoise, benmalpgd_evaluation, spn_evaluation


### setup

# start timer
T0 = time.time()

# init RNG seeds
K0 = 999
K0, K1 = split_key(K0) # shuffle data
K1, K2 = split_key(K1) # init model
K2, K3 = split_key(K2) # global seed set at train time (workaround)


### hyperparameters

# data
FEATURES_RES = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255]).astype('float32')

# architecture
FEATURES_DIM = 10
LABELS_DIM = 5
HIDDEN_DIM = 16
HIDDEN_DEPTH = 4
HIDDEN_ACT = 'relu' # must be str for name formatting

# # gridsearch
# AX0_DOF = 8
# AX1_DOF = AX0_DOF
# AX1_MIN = 1e-16
# AX1_MAX = 0.5

# training
N_NODES = 4
UNIF_LOW = 0.5
UNIF_HIGH = 1.0
N_EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.01

# pgd adversary
PGD_MIN = 0.0
PGD_MAX = 1.0
PGD_RES = 8
PGD_ITER = 7

# spn adversary
SPN_MIN = PGD_MIN
SPN_MAX = PGD_MAX
SPN_RES = PGD_RES
SPN_MAG = 1.0

# tracing
VERBOSE = False


### prepare data

# load partitions
(train_x, train_y), (val_x, val_y), (test_x, test_y) = get_car_hacking_dataset(K1)

# undersample train set
###! train_x must be re-shuffled prior to node partitioning because RUS returns values sorted by class
train_x, train_y = RandomUnderSampler(random_state=K1).fit_resample(train_x, train_y)
train_indices = np.random.RandomState(seed=K1).permutation(len(train_x))
train_x = train_x[train_indices]
train_y = train_y[train_indices]

# generate masks from features
train_mask = np.apply_along_axis(mask_fn, axis=1, arr=train_x)
val_mask = np.apply_along_axis(mask_fn, axis=1, arr=val_x)
test_mask = np.apply_along_axis(mask_fn, axis=1, arr=test_x)

# normalise features
train_x = train_x / FEATURES_RES
val_x = val_x / FEATURES_RES
test_x = test_x / FEATURES_RES

# trace
print(train_x.shape, train_y.shape, 'train')
print(val_x.shape, val_y.shape, 'val')
print(test_x.shape, test_y.shape, 'test')
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### train model

# initialise lambda functions
model_init = lambda:get_multiclass_mlp(
	K2,
	FEATURES_DIM,
	LABELS_DIM,
	HIDDEN_DIM,
	HIDDEN_DEPTH,
	hidden_act=HIDDEN_ACT
)
criterion_init = lambda:losses.SparseCategoricalCrossentropy()
optimizer_init = lambda:optimizers.AdamW(learning_rate=LEARNING_RATE)
metrics_init = lambda:['accuracy']
attack_init = lambda model : BenMalPGD(
	model,
	FEATURES_DIM,
	LABELS_DIM,
	losses.SparseCategoricalCrossentropy(),
	epsilon=PGD_MAX,
	iterations=PGD_ITER,
	batch_size=BATCH_SIZE,
	verbose=VERBOSE
)

###! set global RNG seeds
###! prior to training
np.random.seed(K3)
tf.random.set_seed(K3)

# call train function
train_history, model = federated_uniform_adversarial_train(
	model_init,
	criterion_init,
	optimizer_init,
	metrics_init,
	attack_init,
	FEATURES_RES,
	train_x,
	train_y,
	train_mask,
	val_x,
	val_y,
	val_mask,
	n_nodes=N_NODES,
	unif_lower=UNIF_LOW,
	unif_upper=UNIF_HIGH,
	epochs=N_EPOCHS,
	batch_size=BATCH_SIZE,
	callbacks=None,
	verbose=VERBOSE
)
model.name = f'BIDS_{HIDDEN_DIM}x{HIDDEN_DEPTH}_{HIDDEN_ACT}_Federated_N{N_NODES}_UPGDT_LF{int(1/UNIF_LOW)}_HF{int(1/UNIF_HIGH)}'.replace('.','_')
model.summary()
print(f'[Elapsed time: {time.time()-T0:.2f}s]')


### evaluate model

# create concrete objects
criterion = criterion_init()

# evaluate baseline
test_yh = model.predict(test_x, batch_size=BATCH_SIZE, verbose=int(VERBOSE))
test_loss = criterion(test_y, test_yh).numpy().item()
test_acc = accuracy_score(test_y, np.argmax(test_yh, axis=-1))
test_cfm = confusion_matrix(test_y, np.argmax(test_yh, axis=-1), labels=range(LABELS_DIM))
test_history = dict(
	loss=test_loss,
	accuracy=test_acc,
	confusion=test_cfm
)

# evalute adversarial performance
pgd_history = benmalpgd_evaluation(
	model,
	FEATURES_DIM,
	LABELS_DIM,
	criterion,
	FEATURES_RES,
	test_x,
	test_y,
	test_mask,
	eps_min=PGD_MIN,
	eps_max=PGD_MAX,
	eps_num=PGD_RES,
	pgd_iter=PGD_ITER,
	batch_size=BATCH_SIZE,
	verbose=VERBOSE
)
spn_history = spn_evaluation(
	model,
	criterion,
	LABELS_DIM,
	FEATURES_RES,
	test_x,
	test_y,
	test_mask,
	eps_min=SPN_MIN,
	eps_max=SPN_MAX,
	eps_num=SPN_RES,
	spn_magnitude=SPN_MAG,
	batch_size=BATCH_SIZE,
	verbose=VERBOSE
)

print(train_history)
print(test_history)
print(pgd_history)
print(spn_history)
print(f'[Elapsed time: {time.time()-T0:.2f}s]')
