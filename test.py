from scipy import sparse
import pypianoroll as pp
import numpy as np
import os

max=0
entries=os.listdir(r'E:/GitHub/GANDrum/midi_d/Electronic/')
i=0
for entry in entries:
	i=i+1
	try:
		x=pp.parse(r'E:/GitHub/GANDrum/midi_d/Electronic/'+entry)
		pp.save(r'E:/GitHub/GANDrum/midi_d/your_training_data',x,compressed=False)
		data=np.load(r'E:/GitHub/GANDrum/midi_d/your_training_data.npz')
		mtx=sparse.csc_matrix((data['pianoroll_0_csc_data'], data['pianoroll_0_csc_indices'], data['pianoroll_0_csc_indptr'])).todense()
		if max<mtx.shape[0]:
			max=mtx.shape[0]
			print(max)
			print(i)
	except RuntimeWarning:
		pass

i=0
entries=os.listdir(r'E:/GitHub/GANDrum/midi_d/FunkRnB/')
for entry in entries:
	i=i+1
	try:
		x=pp.parse(r'E:/GitHub/GANDrum/midi_d/FunkRnB/'+entry)
		pp.save(r'E:/GitHub/GANDrum/midi_d/your_training_data',x,compressed=False)
		data=np.load(r'E:/GitHub/GANDrum/midi_d/your_training_data.npz')
		mtx=sparse.csc_matrix((data['pianoroll_0_csc_data'], data['pianoroll_0_csc_indices'], data['pianoroll_0_csc_indptr'])).todense()
		if max<mtx.shape[0]:
			max=mtx.shape[0]
			print(max)
			print(i)
	except RuntimeWarning:
		pass
i=0
entries=os.listdir(r'E:/GitHub/GANDrum/midi_d/World/')
for entry in entries:
	i=i+1
	try:
		x=pp.parse(r'E:/GitHub/GANDrum/midi_d/World/'+entry)
		pp.save(r'E:/GitHub/GANDrum/midi_d/your_training_data',x,compressed=False)
		data=np.load(r'E:/GitHub/GANDrum/midi_d/your_training_data.npz')
		mtx=sparse.csc_matrix((data['pianoroll_0_csc_data'], data['pianoroll_0_csc_indices'], data['pianoroll_0_csc_indptr'])).todense()
		if max<mtx.shape[0]:
			max=mtx.shape[0]
			print(max)
			print(i)
	except RuntimeWarning:
		pass
		




