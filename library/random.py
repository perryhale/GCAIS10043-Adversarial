###! no imports

# split key n times with quadratic function
# type: (int, int, int, bool) -> List[int]
def split_key(key, n=2, mod=10**9, verbose=False):
	if key==0: print('Warning: using zero key')
	keys = [int((i*key**2) % mod) for i in range(1,1+n)]
	if verbose: print(keys)
	return keys
