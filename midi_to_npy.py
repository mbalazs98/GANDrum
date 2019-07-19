import numpy as np
import pypianoroll as pp
from matplotlib import pyplot as plt
from scipy import sparse

x=pp.parse(r'D:\midi_drum\contemporary\Contemporary4.mid')
pp.save(r'D:/MidiNet/v1/your_training_data',x,compressed=False)

#print(len(data['pianoroll_0_csc_indptr']))

data = np.load(r'D:\MidiNet\v1\your_training_data.npz')

mtx = sparse.csc_matrix((data['pianoroll_0_csc_data'], data['pianoroll_0_csc_indices'], data['pianoroll_0_csc_indptr']), shape=(1536, 128)).todense()
print(mtx)
'''
x=parse(r'D:\midi_drum\Latin\Latin2.mid')
save(r'D:/MidiNet/v1/your_training_data',x,compressed=False)
data = np.load(r'D:/MidiNet/v1/your_training_data.npz')
lst = np.asarray(data.files)
print(type(lst[0]))

track = Track(pianoroll=lst, program=0, is_drum=True, name='my awesome drum')
			  
# Plot the piano-roll
fig, ax = track.plot()
plt.show()
'''