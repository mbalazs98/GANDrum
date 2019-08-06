from scipy import sparse
import pypianoroll as pp
import numpy as np

entries=os.listdir(r'E:/GitHub/GANDrum/midi_d/Electronic/')
j=0
for entry in entries:
	j=j+1

	x=pp.parse(r'E:/GitHub/GANDrum/midi_d/Electronic/(1).mid')
	pp.save(r'E:/GitHub/GANDrum/your_training_data',x,compressed=False)
	data=np.load(r'E:/GitHub/GANDrum/your_training_data.npz')
	mtx=sparse.csc_matrix((data['pianoroll_0_csc_data'], data['pianoroll_0_csc_indices'], data['pianoroll_0_csc_indptr'])).todense()
	tempo=data['tempo']
	downbeat=data['downbeat']


	for i in range (0,int(mtx.shape[0]/96)):
		
		curr_mtx=mtx[i*96:(i+1)*96,:]
		track=[pp.Track(pianoroll=curr_mtx,is_drum=True)]
		multitrack=pp.Multitrack(tracks=track,tempo=tempo,downbeat=downbeat)
		pp.write(multitrack,r'E:/GitHub/GANDrum/{}.mid'.format(i))
	

