import numpy as np

labels = ['contemporary','dance','ethnic','three/four','folk','jazz','latin','rock']
y = []
for name in labels:
	for i in range(0,10):
		y.append(name)
mat = np.array(y)

np.save('D:\midi_d_label\labels',mat)