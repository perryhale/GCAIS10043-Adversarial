###! no imports

# type: (int, int) -> List[int]
def split_key(key, n=2):
	if key==0: print('Warning: using zero key')
	return [key * (i+1) for i in range(n)]
