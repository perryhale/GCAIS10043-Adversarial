import os
import random
import numpy as np
import tensorflow as tf

# split key n times with quadratic function
# type: (int, int, int, bool) -> List[int]
def split_key(key, n=2, mod=10**9, verbose=False):
	if key==0: print('Warning: using zero key')
	keys = [int((i*key**2) % mod) for i in range(1,1+n)]
	if verbose: print(keys)
	return keys

# type: (int) -> None
def seed_everything(key):
    random.seed(key)
    np.random.seed(key)
    tf.random.set_seed(key)
    os.environ['PYTHONHASHSEED'] = str(key)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
