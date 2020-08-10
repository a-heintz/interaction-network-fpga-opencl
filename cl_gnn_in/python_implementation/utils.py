import os
from models.graph import Graph, load_graph
from torch.autograd import Variable
import torch

def get_graphs(d):
    """ get all graphs from a directory
          return [("event<id1>", graph1, size), ...]
    """
    files = os.listdir(d)
    return [load_graph(d+f) for f in files]

def get_inputs(graphs):
    size = len(graphs)
    O = [Variable(torch.FloatTensor(graphs[i].X))
         for i in range(size)]
    Rs = [Variable(torch.FloatTensor(graphs[i].Ro))
          for i in range(size)]
    Rr = [Variable(torch.FloatTensor(graphs[i].Ri))
          for i in range(size)]
    Ra = [Variable(torch.FloatTensor(graphs[i].a)).unsqueeze(0)
          for i in range(size)]
    y  = [Variable(torch.FloatTensor(graphs[i].y)).unsqueeze(0).t()
          for i in range(size)]
    return O, Rs, Rr, Ra, y