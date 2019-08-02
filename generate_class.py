import numpy as np

y = []

for i in range(0,1089):
	y.append('Electronic')

for i in range(0,656):
	y.append('Funk')

for i in range(0,926):
	y.append('World')
	
mat = np.array(y)

np.save('D:\midi_d_label\labels',mat)