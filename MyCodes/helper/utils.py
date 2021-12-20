import os
from models.structure_model.tree import Tree


def load_hierarchy_label_file(hierarchy_label_file, label_map):
    assert os.path.isfile(hierarchy_label_file)
    hierarchy_label_dict = dict()
    label_tree = dict()
    root = Tree(-1)    # 根节点
    with open(hierarchy_label_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            labels = line.rstrip('\n').split('\t')
            parent, children = labels[0], labels[1:]
            if parent not in label_map.keys():
                if parent == 'Root':
                    hierarchy_label_dict[parent] = -1
                    label_tree[parent] = root
                else:
                    continue
            else:
                hierarchy_label_dict[parent] = label_map[parent]
                for child_vocab in children:
                    if child in label_map:
                        child = Tree(hierarchy_label_dict[child_vocab])
                        label_tree[parent].add_child(child)
    return hierarchy_label_dict, label_tree