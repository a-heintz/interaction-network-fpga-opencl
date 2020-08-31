import os
import logging
import math
import sys
sys.path.append("../")
sys.path.append("../trackml-library/")

import torch
import numpy as np
import pandas as pd

import trackml
from trackml.dataset import load_event
from trackml.dataset import load_dataset
from Model.Graph import Graph
import visualization_scripts.plot_functions as pf

verbose = False
evt_num = "event000001000"

print("Processing", evt_num)
hits, cells, particles, truth = load_event(os.path.join('../data', evt_num))

evtid = int(evt_num[-9:])

pixel_layers = [(8,2), (8,4), (8,6), (8,8)]
hits_by_loc = hits.groupby(['volume_id', 'layer_id'])
hits = pd.concat([hits_by_loc.get_group(pixel_layers[i]).assign(layer=i)
                  for i in range(len(pixel_layers))])

pt = np.sqrt(particles.px**2 + particles.py**2)
particles = particles[pt > 1]
if (verbose): print("Keeping particles:", particles.head(200))

truth = (truth[['hit_id', 'particle_id', 'weight']]
         .merge(particles[['particle_id']], on='particle_id'))

r = np.sqrt(hits.x**2 + hits.y**2)
phi = np.arctan2(hits.y, hits.x)

hits = (hits[['hit_id', 'x', 'y', 'z', 'layer']]
        .assign(r=r, phi=phi)
        .merge(truth[['hit_id', 'particle_id', 'weight']], on='hit_id'))
hits = hits.assign(evtid=evtid)
if (verbose): print("Keeping hits:", hits.head(200))

n_hits = hits.shape[0]

feature_names = ['r', 'phi', 'z']
feature_scale = np.array([1000., np.pi, 1000.])

n_layers = len(pixel_layers)
l = np.arange(n_layers)
layer_pairs = np.stack([l[:-1], l[1:]], axis=1)
layer_pairs = np.concatenate((np.stack([l, l], axis=1), np.stack([l[:-1], l[1:]], axis=1)), axis=0)
layer_groups = hits.groupby('layer')

# kNN clustering
hit_tensor = torch.tensor(hits[['x', 'y', 'z']].values)
dists = torch.empty(n_hits, n_hits)
for i in range(n_hits):
    for j in range(n_hits):
        dists[i][j] = torch.norm(hit_tensor[i]-hit_tensor[j])

segments = []
a = []

# layer pairing code
dR_values = []
dR_values_diff = []
same_pid = []
for (layer1, layer2) in layer_pairs:
    if (verbose): print("layer1, layer2=", layer1, layer2)
    try:
        hits1=layer_groups.get_group(layer1)
        hits2=layer_groups.get_group(layer2)
    except KeyError as e:
        continue
        
    keys = ['evtid', 'r', 'phi', 'z', 'particle_id']
    hit_pairs = hits1[keys].reset_index().merge(
        hits2[keys].reset_index(), on='evtid', suffixes=('_1', '_2'))
    hit_pairs = hit_pairs[hit_pairs.index_1 != hit_pairs.index_2]
    if (verbose): print("Adding hit_pairs:", hit_pairs[['index_1', 'index_2']].head(200))
    
    dz = hit_pairs.z_2 - hit_pairs.z_1
    dr = hit_pairs.r_2 - hit_pairs.r_1
    
    if(layer1 == layer2):
        dR = np.sqrt(dr**2 + dz**2).to_numpy()
        pid1 = hit_pairs.particle_id_1.to_numpy()
        pid2 = hit_pairs.particle_id_2.to_numpy()
        for i in range(dR.shape[0]):
            if (pid1[i] == pid2[i]): 
                same_pid.append(1)
                dR_values.append(dR[i])
                print("Same particle:", dR[i])
            else:
                same_pid.append(0)
                dR_values_diff.append(dR[i])
                print("Opposite particle:", dR[i])
      
    for i in range(hit_pairs.shape[0]):
        if (layer1==layer2): 
            a.append(1)
        else: a.append(0)
            
    segments.append(hit_pairs)


pf.plotSingleHist(dR_values, 'dR', 'Counts', 20, title='Same Layer, Same Particle')
pf.plotDoubleHistOverlapped(dR_values, dR_values_diff, 'dR', 'Counts', 
                            50, 0, 200, label1="Same Particle", label2="Different Particles", 
                            figLabel="dR_Comparison.png", saveFig=True)

segments = pd.concat(segments)
n_hits = hits.shape[0]
n_edges = segments.shape[0]

X = (hits[feature_names].values / feature_scale).astype(np.float32)
Ri = np.zeros((n_hits, n_edges), dtype = np.uint8)
Ro = np.zeros((n_hits, n_edges), dtype = np.uint8)
y = np.zeros(n_edges, dtype=np.float32)

hit_idx = pd.Series(np.arange(n_hits), index=hits.index)
seg_start = hit_idx.loc[segments.index_1].values
seg_end = hit_idx.loc[segments.index_2].values
if (verbose): print("seg_start", seg_start)
if (verbose): print("seg_end", seg_end)
if (verbose): print("adjacencies", a)

Ri[seg_end, np.arange(n_edges)] = 1
Ro[seg_start, np.arange(n_edges)] = 1

pid1 = hits.particle_id.loc[segments.index_1].values
pid2 = hits.particle_id.loc[segments.index_2].values
y[:] = (pid1==pid2)

g = Graph(X, Ri, Ro, y, a)
if (verbose): print(g)

print(hits.head(10))
