class Tree(object):
    def __init__(self, idx):
        super(Tree, self).__init__()
        self.idx = idx
        self.bottom_up_state = None
        self.top_down_state = None
        self.num_children = 0
        self.parent = 0
        self.children = list()
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        self.num_children += 1