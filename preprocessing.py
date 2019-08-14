from scipy import sparse
import pypianoroll as pp
import numpy as np
import os

entries=os.listdir(r'D:\GANDrum\midi_d\World')
j=1
for entry in entries:
	

	data=pp.parse(r'D:/GANDrum/midi_d/World/'+entry)
	tempo=data.tempo
	downbeat=data.downbeat


	for i in range (0,int(data.tracks[0].pianoroll.shape[0]/96)):
		
		curr_mtx=data.tracks[0].pianoroll[i*96:(i+1)*96,:]
		track=[pp.Track(pianoroll=curr_mtx,is_drum=True)]
		multitrack=pp.Multitrack(tracks=track,tempo=tempo,downbeat=downbeat)
		pp.write(multitrack,r'D:\GANDrum\midi_d_processed\World{}.mid'.format(j))
		j=j+1