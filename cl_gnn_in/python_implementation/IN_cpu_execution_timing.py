#!/usr/bin/env python
# coding: utf-8

# In[1]:


from models.graph import Graph, load_graph
from models.interaction_network import InteractionNetwork
from utils import *
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import h5py

prep, pt_cut = 'LP', 5
graph_dir = "../IN_{0}_{1}/".format(prep, pt_cut)
model_path = './trained_models/LP_2_1500_wide_noPhi_epoch300.pt'
data_path = '../data'

graphs = get_graphs(graph_dir)
print('length of graphs: ', len(graphs))
objects, sender_relations, receiver_relations, relation_info, y = get_inputs(graphs)

weights = torch.load(model_path)

object_dim, relation_dim, effect_dim = 3, 1, 1
interaction_network = InteractionNetwork(object_dim, relation_dim, effect_dim)

interaction_network.load_state_dict(weights)
interaction_network.eval()
start = time.time()
pred = interaction_network(objects, sender_relations, receiver_relations, relation_info)
end = time.time()
pytorch_pred = torch.cat(pred, dim=0)
target = torch.cat(y, dim=0)
print(pytorch_pred.shape, target.shape)
print(end - start, ' seconds')


# In[ ]:
