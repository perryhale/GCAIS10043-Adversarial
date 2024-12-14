import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import plot_tree


### set RNG seed
RNG_SEED = 0
np.random.seed(RNG_SEED)


### data preparation
TRAIN_RATIO = 0.7
VAL_RATIO = 0.05
TEST_RATIO = 1 - (TRAIN_RATIO + VAL_RATIO)
FEATURE_SCALE = np.array([4095, 8, 255, 255, 255, 255, 255, 255, 255, 255])
BATCH_SIZE = 512

# type: (pd.DataFrame, float, float) -> tuple[pd.DataFrame]
def train_val_test_split(data):
	train_data = data.iloc[:int(TRAIN_RATIO*len(data.index)), :]
	val_data = data.iloc[int(TRAIN_RATIO*len(data.index)):int((TRAIN_RATIO+VAL_RATIO)*len(data.index)), :]
	test_data = data.iloc[int((TRAIN_RATIO+VAL_RATIO)*len(data.index)):, :]
	return (train_data, val_data, test_data)

# type: (pd.DataFrame) -> tuple[np.ndarray]
def standard_split(data):
	x = data.iloc[:, :-1].to_numpy() / FEATURE_SCALE
	y = data.iloc[:, -1].to_numpy()
	return (x, y)

# load data and shuffle
data = pd.read_csv('car_hacking_dataset/car_hacking_dataset.csv', header=None)
data = data.sample(frac=1)

# basic partition
train_data, val_data, test_data = train_val_test_split(data)

# train set oversample and standard split
train_x, train_y = standard_split(train_data)
train_x, train_y = RandomOverSampler().fit_resample(train_x, train_y)

# val and test standard split 
val_x, val_y = standard_split(val_data)
test_x, test_y = standard_split(test_data)

print(train_x.shape, train_y.shape, 'train')
print(val_x.shape, val_y.shape, 'validation')
print(test_x.shape, test_y.shape, 'test')


### define model
model = RandomForestClassifier(n_estimators=10, max_depth=5) # uses numpy RNG backend
model.fit(train_x, train_y)
# with open('decisiontree_ids_sklearn.pkl', 'rb') as file:
# 	model = pickle.load(file)


### evaluate model
test_yh = model.predict(test_x)
test_accuracy = accuracy_score(test_y, test_yh)
test_cfm = confusion_matrix(test_y, test_yh)

print(f'Test accuracy: {test_accuracy}')
print(test_cfm)


### save results

# save evaluation scores
with open('randomforest_ids_sklearn.test.txt', 'w') as file:
	file.write(f'Test accuracy: {test_accuracy}\n')
	file.write(f'{str(test_cfm)}\n')

# save model
with open('randomforest_ids_sklearn.pkl', 'wb') as file:
	pickle.dump(model, file)