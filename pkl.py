import numpy as np
import pandas as pd
import os

N = 100  # matrix dimension
feature_dim = 5
inputdir = '../train_x_beta1to10_i100_j0/'
outputdir = '../train_y_beta1to10_i100_j0/'
num_files = int(len(os.listdir(inputdir)))  # number of files
print(num_files)
columns = ['x'+str(i) for i in range(N)]
columns += ['y'+str(i) for i in range(feature_dim)]
# print(columns)

data_complete = []
for i in range(num_files):
	fi = open(inputdir+'/x_'+str(i))
	content = fi.readlines()
	#tot_row = content[0]
	#assert (int(tot_row) == N)
	data = [0 for i in range(N+feature_dim)]
	#data = [float(line) for line in content[0:]]
	nn = 0
	for line in content[0:]:
		data[nn] = float(line)
		nn = nn+1

	fi = open(outputdir+'/y_'+str(i))
	content = fi.readlines()
	#tot_row = content[0]
	#assert (int(tot_row) == N)
	#data += [float(line) for line in content[0:]]
	#data.append([0 for i in range(feature_dim)])	
	print(len(data))
	data[N+int(content[0][0])] = 1

	data_complete.append(data)

data_complete = np.array(data_complete)
print(data_complete.shape)
df = pd.DataFrame(data=data_complete,columns=columns)
print(df.head())

df.to_pickle('./train_dataset_beta1to10_i100_j0.pkl')
os.system('git add ./train_dataset_beta1to10_i100_j0.pkl')
os.system('git add ./pkl.py')
os.system('git commit')
os.system('git push')


