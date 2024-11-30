import numpy as np
import re
from tqdm import tqdm

with open('normal_run_data.txt', 'r') as f:
	lines = f.readlines()[:-2]

extract = []
for l in tqdm(lines):
	regex = r"Timestamp:\s*(\d+\.\d+)\s+ID:\s*([0-9a-fA-F]+)\s*\d*\s*DLC:\s*(\d+)\s*([0-9A-Fa-f\s]+)\s*"
	match = re.search(regex, l.replace('\n',''))
	timestamp = match.group(1)
	identifier = match.group(2)
	dlc = match.group(3)
	data = match.group(4).split(' ')
	row = [str(timestamp), str(identifier), str(dlc), *data, 'R']
	extract.append(row)

with open('../normal_dataset.csv', 'w') as f:
	for l in extract:
		f.write(','.join(l) + '\n')
