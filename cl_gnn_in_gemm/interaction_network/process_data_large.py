#!/usr/bin/env python
import argparse
from models.graph import Graph, load_graph
from models.interaction_network import InteractionNetwork
from utils import *
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import h5py
import yaml

data_path = '/tigress/aheintz/data'

args = parse_args()
with open(args.config) as f:
    config = yaml.load(f, yaml.FullLoader)

model_outdir = config['model_outdir']
plot_outdir  = config['plot_outdir']
verbose = config['verbose']
prep, pt_cut = config['prep'], config['pt_cut']
n_epoch, batch_size = config['n_epoch'], config['batch_size']
save_every = config['save_every_n_epoch']
save_last  = config['save_last_n_epoch']
phi_reflect = config['phi_reflect']
tag = config['tag']
graph_dir = config['graph_dir']

job_name = "{0}_{1}_{2}_{3}".format(prep, pt_cut, n_epoch, tag)
model_path = '{}/{}_epoch0.pt'.format(model_outdir, job_name)

graphs = get_graphs(graph_dir)
print(len(graphs))
objects, sender_relations, receiver_relations, relation_info, y = get_inputs(graphs)

weights = torch.load(model_path)
# In[2]:
#object_dim, relation_dim, effect_dim = 3, 1, 1
#interaction_network = InteractionNetwork(object_dim, relation_dim, effect_dim)

#interaction_network.load_state_dict(weights)
#print(interaction_network.eval())

# In[3]:
#start = time.time()
#pred = interaction_network(objects, sender_relations, receiver_relations, relation_info)
#end = time.time()
#pytorch_pred = torch.cat(pred, dim=0)
#target = torch.cat(y, dim=0)
#print(pytorch_pred.shape, target.shape, end - start)

# In[5]:


hf = h5py.File('{}/test_{}_{}.hdf5'.format(data_path, prep, pt_cut), 'w')
obj_ms = []
sr_ms = []
rr_ms = []
ri_ms = []
obj_ns = []
sr_ns = []
rr_ns = []
ri_ns = []

for i in range(len(graphs)):
    obj = objects[i].numpy()#.tolist()
    sr = sender_relations[i].numpy()#.tolist()
    rr = receiver_relations[i].numpy()#.tolist()
    ri = relation_info[i].numpy()#.tolist()
    hf.create_dataset('obj_{}'.format(i), data=obj.tolist())
    hf.create_dataset('sr_{}'.format(i), data=sr.tolist())
    hf.create_dataset('rr_{}'.format(i), data=rr.tolist())
    hf.create_dataset('ri_{}'.format(i), data=ri.tolist())

    obj_ms.append(obj.shape[0])
    sr_ms.append(sr.shape[0])
    rr_ms.append(rr.shape[0])
    ri_ms.append(ri.shape[0])

    obj_ns.append(obj.shape[1])
    sr_ns.append(sr.shape[1])
    rr_ns.append(rr.shape[1])
    ri_ns.append(ri.shape[1])

hf.create_dataset('obj_shape_0_i', data=obj_ms)
hf.create_dataset('sr_shape_0_i', data=sr_ms)
hf.create_dataset('rr_shape_0_i', data=rr_ms)
hf.create_dataset('ri_shape_0_i', data=ri_ms)

hf.create_dataset('obj_shape_1_i', data=obj_ns)
hf.create_dataset('sr_shape_1_i', data=sr_ns)
hf.create_dataset('rr_shape_1_i', data=rr_ns)
hf.create_dataset('ri_shape_1_i', data=ri_ns)
hf.close()
hf = h5py.File('{}/model_weights_{}_{}.hdf5'.format(data_path, prep, pt_cut), 'w')
for k in weights.keys():
    weight = weights[k].t()
    hf.create_dataset(k, data=weight.numpy().tolist())
hf.close()

# In[6]:


print(len(objects), len(sender_relations), len(receiver_relations), len(relation_info))
print('done with {}.'.format(job_name))
# In[ ]:
