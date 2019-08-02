from scipy import sparse
import pypianoroll as pp
import numpy as np
import os

max=0

entries=os.listdir(r'D:/midi_d/Electronic/')
for entry in entries:
	x=pp.parse(entry)
	pp.save(r'D:\midi_d\your_training_data',x,compressed=False)
	data=np.load(r'D:\midi_d\your_training_data.npz')
	mtx=sparse.csc_matrix((data['pianoroll_0_csc_data'], data['pianoroll_0_csc_indices'], data['pianoroll_0_csc_indptr'])).todense()
	if max<mtx.shape[0]:
		max=mtx.shape[0]
		print(max)

entries=os.listdir(r'D:/midi_d/FunkRnB/')
for entry in entries:
	x=pp.parse(entry)
	pp.save(r'D:\midi_d\your_training_data',x,compressed=False)
	data=np.load(r'D:\midi_d\your_training_data.npz')
	mtx=sparse.csc_matrix((data['pianoroll_0_csc_data'], data['pianoroll_0_csc_indices'], data['pianoroll_0_csc_indptr'])).todense()
	if max<mtx.shape[0]:
		max=mtx.shape[0]
		print(max)

entries=os.listdir(r'D:/midi_d/World/')
for entry in entries:
	x=pp.parse(entry)
	pp.save(r'D:\midi_d\your_training_data',x,compressed=False)
	data=np.load(r'D:\midi_d\your_training_data.npz')
	mtx=sparse.csc_matrix((data['pianoroll_0_csc_data'], data['pianoroll_0_csc_indices'], data['pianoroll_0_csc_indptr'])).todense()
	if max<mtx.shape[0]:
		max=mtx.shape[0]
		print(max)
		




