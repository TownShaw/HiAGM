import os
import json
from posixpath import pardir
import numpy as np
import torch
import torch.nn as nn
from models.structure_model.tree import Tree
from models.structure_model.weighted_tree_lstm import WeightedHierarchicalTreeLSTMEndtoEnd
from models.structure_model.graphcnn import HierarchyGCN


MODEL_MODULE = {
    'TreeLSTM': WeightedHierarchicalTreeLSTMEndtoEnd,
    'GCN': HierarchyGCN
}

class StructureEncoder(nn.Module):
    def __init__(self,
                 config,
                 label_map,
                 device,
                 graph_model_type):
        super(StructureEncoder, self).__init__()
        self.label_map = label_map
        self.config = config
        hierarchy_label_file = os.path.join(self.config.dict['data']['data_dir'], self.config.dict['data']['hierarchy'])

        self.load_hierarchy_label_file(hierarchy_label_file)

        hierarchy_prob_file = os.path.join(self.config.dict['data']['data_dir'], self.config.dict['data']['prob_json'])
        with open(hierarchy_prob_file, 'r', encoding='utf-8') as fin:
            line = fin.readline()
            self.hierarchy_prob = json.loads(line.rstrip())
        
        self.node_prob_from_parent = np.zeros((len(self.label_map), len(self.label_map)))
        self.node_prob_from_child = np.zeros((len(self.label_map), len(self.label_map)))

        for parent in self.hierarchy_prob.keys():
            if parent.lower() == "root":
                continue
            else:
                for child in self.hierarchy_prob[parent].keys():
                    parent = parent.lower()
                    child = child.lower()
                    self.node_prob_from_child[int(self.label_map[parent])][int(self.label_map[child])] = 1.0
                    self.node_prob_from_parent[int(self.label_map[child])][int(self.label_map[parent])] = self.hierarchy_prob[parent.upper()][child.upper()]
        
        self.model = MODEL_MODULE[graph_model_type](num_nodes=len(self.label_map),
                                                    in_matrix=self.node_prob_from_child,
                                                    out_matrix=self.node_prob_from_parent,
                                                    in_dim=self.config.dict['structure_encoder']['node']['dimension'],
                                                    dropout=self.config.dict['structure_encoder']['node']['dropout'],
                                                    device=device,
                                                    root=self.root,
                                                    label_tree=self.label_tree,
                                                    hierarchical_label_dict=self.hierarchy_label_dict)


    def load_hierarchy_label_file(self, hierarchy_label_file):
        assert os.path.isfile(hierarchy_label_file)
        self.hierarchy_label_dict = dict()
        self.label_tree = dict()
        self.root = Tree(-1)    # 根节点
        self.hierarchy_label_dict['root'] = -1
        self.label_tree['root'] = self.root
        with open(hierarchy_label_file, 'r', encoding='utf-8') as fin:
            for line in fin:
                labels = line.rstrip('\n').split('\t')
                parent, children = labels[0].lower(), labels[1:]
                if parent not in self.label_map.keys() and parent != 'root':
                    continue
                else:
                    # self.hierarchy_label_dict[parent] = self.label_map[parent]
                    for child_vocab in children:
                        child_vocab = child_vocab.lower()
                        if child_vocab in self.label_map:
                            self.hierarchy_label_dict[child_vocab] = self.label_map[child_vocab]
                            child = Tree(self.hierarchy_label_dict[child_vocab])
                            self.label_tree[child_vocab] = child
                            self.label_tree[parent].add_child(child)

    def forward(self, inputs):
        return self.model(inputs)