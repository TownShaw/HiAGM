import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HierarchyGCN(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_matrix,
                 out_matrix,
                 in_dim,
                 dropout=0.0,
                 device=torch.device('cpu'),
                 root=None,
                 hierarchical_label_dict=None,
                 label_trees=None):
        """
        Graph Convolutional Network variant for hierarchy structure
        original GCN paper:
                Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks.
                    arXiv preprint arXiv:1609.02907.
        :param num_nodes: int, N
        :param in_matrix: numpy.Array(N, N), input adjacent matrix for child2parent (bottom-up manner)
        :param out_matrix: numpy.Array(N, N), output adjacent matrix for parent2child (top-down manner)
        :param in_dim: int, the dimension of each node <- config.structure_encoder.node.dimension
        :param layers: int, the number of layers <- config.structure_encoder.num_layer
        :param time_step: int, the number of time steps <- config.structure_encoder.time_step
        :param dropout: Float, P value for dropout module <- configure.structure_encoder.node.dropout
        :param prob_train: Boolean, train the probability matrix if True <- config.structure_encoder.prob_train
        :param device: torch.device <- config.train.device_setting.device
        """
        super(HierarchyGCN, self).__init__()

class HierarchyGCNModule(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_adj, out_adj,
                 in_dim, dropout, device, in_arc=True, out_arc=True,
                 self_loop=True):
        super(HierarchyGCNModule, self).__init__()