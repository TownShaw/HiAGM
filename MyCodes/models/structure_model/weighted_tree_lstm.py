from models.structure_model.tree import Tree
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class WeightedHierarchicalTreeLSTMEndtoEnd(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_matrix,
                 out_matrix,
                 in_dim,
                 dropout=0.0,
                 device=torch.device('cpu'),
                 root='Root',
                 label_tree=None,
                 hierarchical_label_dict=None
                 ):
        super(WeightedHierarchicalTreeLSTMEndtoEnd, self).__init__()
        self.mem_dim = in_dim // 2
        self.root = root
        self.label_tree = label_tree
        self.hierarchy_label_dict = hierarchical_label_dict
        self.bottom_up_lstm = WeightedChildSumTreeLSTMEndtoEnd(in_dim, self.mem_dim, label_tree, root=self.root, prob=in_matrix)
        self.top_down_lstm = WeightedTopDownTreeLSTMEndtoEnd(in_dim, self.mem_dim, label_tree, root=self.root, prob=out_matrix)
        self.tree_feature = nn.Linear(self.mem_dim * 2, self.mem_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs):
        inputs = inputs.transpose(0, 1)  # N, batch_size, dim
        for label in self.hierarchy_label_dict.keys():
            self.bottom_up_lstm(self.label_tree[label], inputs)
            self.top_down_lstm(self.label_tree[label], inputs)
        tree_feature = []
        for label in self.hierarchy_label_dict.keys():
            if label == 'root':
                continue
            tree_feature.append(torch.cat((self.dropout(self.label_tree[label].bottom_up_state[1].view(inputs.shape[1], 1, self.mem_dim)),
                                          self.dropout(self.label_tree[label].top_down_state[1].view(inputs.shape[1], 1, self.mem_dim))), 2))
        label_feature = torch.cat(tree_feature, 1)  # label_feature: batch_size, num_nodes, 2 * node_dimension

        return label_feature

class WeightedChildSumTreeLSTMEndtoEnd(nn.Module):
    def __init__(self,
                 in_dim,
                 mem_dim,
                 label_tree=None,
                 root=None,
                 prob=None,
                 device=torch.device('cpu')):
        super(WeightedChildSumTreeLSTMEndtoEnd, self).__init__()
        self.root = root
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.label_tree = label_tree
        self.ioux = nn.Linear(in_dim, 3 * mem_dim)
        self.iouh = nn.Linear(mem_dim, 3 * mem_dim)
        self.fx = nn.Linear(in_dim, mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim)
        self.prob = torch.Tensor(prob).to(device)
        self.prob = Parameter(self.prob)

    def node_forward(self, inputs, child_c, child_h):
        # print(inputs.size(), child_c.size(), child_h.size())
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)
        # print(child_h_sum.size())
        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(2) // 3, dim=2)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(self.fh(child_h) + self.fx(inputs).repeat(len(child_h), 1, 1))
        # print(f.size())
        fc = torch.mul(f, child_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, root, inputs):
        for child in root.children:
            self.forward(child, inputs)
        
        if root.num_children == 0:
            child_c = inputs[0, 0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_().repeat(1, inputs.shape[1],
                                                                                                   1)
            child_h = inputs[0, 0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_().repeat(1, inputs.shape[1],
                                                                                                   1)
        else:
            child_c, child_h = zip(
                *map(lambda x: (self.prob[root.idx][x.idx] * y for y in x.bottom_up_state), root.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)
        root.bottom_up_state = self.node_forward(inputs[root.idx], child_c, child_h)
        return root.bottom_up_state


class WeightedTopDownTreeLSTMEndtoEnd(nn.Module):
    def __init__(self,
                 in_dim,
                 mem_dim,
                 label_tree=None,
                 root=None,
                 prob=None,
                 device=torch.device('cpu')):
        super(WeightedTopDownTreeLSTMEndtoEnd, self).__init__()
        self.root = root
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.label_tree = label_tree
        self.ioux = nn.Linear(in_dim, 3 * mem_dim)
        self.iouh = nn.Linear(mem_dim, 3 * mem_dim)
        self.fx = nn.Linear(in_dim, mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim)
        self.prob = torch.Tensor(prob).to(device)
        self.prob = Parameter(self.prob)

    def node_forward(self, inputs, parent_c, parent_h):

        iou = self.ioux(inputs) + self.iouh(parent_h)
        i, o, u = torch.split(iou, iou.size(2) // 3, dim=2)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(self.fh(parent_h) + self.fx(inputs).repeat(len(parent_h), 1, 1))
        fc = torch.mul(f, parent_c)
        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, root, inputs, state=None, parent=None):
        if state is None:
            parent_c = inputs[0, 0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_().repeat(1, inputs.shape[1],
                                                                                                   1)
            parent_h = inputs[0, 0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_().repeat(1, inputs.shape[1],
                                                                                                   1)
        else:
            parent_c = self.prob[parent.idx][root.idx] * state[0]
            parent_h = self.prob[parent.idx][root.idx] * state[1]
        root.top_down_state = self.node_forward(inputs[root.idx], parent_c, parent_h)
        for idx in range(root.num_children):
            self.forward(root.children[idx], inputs, root.top_down_state, root)
        return root.top_down_state