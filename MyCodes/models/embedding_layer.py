import os
import torch
import torch.nn as nn
from torch.nn.modules.sparse import Embedding
import helper.logger as logger
from torch.nn.init import uniform_

class EmbeddingLayer(nn.Module):
    def __init__(self,
                 vocab_map,
                 embedding_dim,
                 vocab_name,
                 config,
                 padding_index=None,
                 pretrained_dir=None,
                 model_mode='TRAIN',
                 initial_type='kaiming_uniform',
                 negative_slope=0, mode_fan='fan_in',
                 activation_type='linear',
                 ):
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(p=config.dict['embedding'][vocab_name]['dropout'])
        self.embedding = nn.Embedding(len(vocab_map), embedding_dim, padding_index)

        self.lookup_table = uniform_(torch.empty(len(vocab_map), embedding_dim), a=-0.25, b=0.25)

        if pretrained_dir is not None and os.path.isfile(pretrained_dir) \
                and model_mode == 'TRAIN' and config.dict['embedding'][vocab_name]['type'] == 'pretrain':
            self.load_from_pretrained_file(pretrained_dir, embedding_dim, vocab_name, vocab_map)
        
        if padding_index is not None:
            self.lookup_table[padding_index] = torch.FloatTensor([0. for i in range(embedding_dim)])
        self.embedding.weight.data.copy_(self.lookup_table)
        self.embedding.weight.data.requires_grad_(True)
        del self.lookup_table


    def load_from_pretrained_file(self, pretrained_dir, embedding_dim, vocab_name, vocab_map):
        with open(pretrained_dir, "r", encoding="utf-8") as fin:
            num_pretrained = 0
            for line in fin:
                data = line.rstrip('\n').split(' ')
                if data[0] in vocab_map.keys():
                    self.lookup_table[vocab_map[data[0]]] = torch.FloatTensor([float(i) for i in data[1:]])
                    num_pretrained += 1
        logger.info('Total vocab size of %s is %d.' % (vocab_name, len(vocab_map)))
        logger.info('Pretrained vocab embedding has %d / %d' % (num_pretrained, len(vocab_map)))
    
    def forward(self, vocab_id_list):
        vocab_embedding = self.embedding(vocab_id_list)
        return self.dropout(vocab_embedding)