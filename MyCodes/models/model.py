import torch
import torch.nn as nn
from models.embedding_layer import EmbeddingLayer
from models.text_encoder import TextEncoder
from models.structure_model.structure_encoder import StructureEncoder
from models.text_feature_propagation import HiAGM_TP

DATAFLOW_TYPE = {
    'HiAGM-TP': 'serial',
    'HiAGM-LA': 'parallel',
    'Origin': 'origin'
}

class HiAGM(nn.Module):
    def __init__(self, config, vocab, model_type, model_mode='TRAIN'):
        super(HiAGM, self).__init__()
        self.config = config
        self.vocab = vocab
        self.device = config.dict['train']['device_setting']['device']

        self.token_map, self.label_map = vocab.v2i['token'], vocab.v2i['label']
        self.embedding = EmbeddingLayer(vocab_map=self.token_map,
                                        embedding_dim=config.dict['embedding']['token']['dimension'],
                                        vocab_name='token',
                                        config=config,
                                        padding_index=vocab.padding_index,
                                        pretrained_dir=config.dict['embedding']['token']['pretrained_file'])

        self.text_encoder = TextEncoder(config)
        self.structure_encoder = StructureEncoder(config,
                                                  label_map=self.label_map,
                                                  device=self.device,
                                                  graph_model_type=config.dict['structure_encoder']['type'])

        self.hiagm = HiAGM_TP(config=config,
                              device=self.device,
                              graph_model=self.structure_encoder,
                              label_map=self.label_map)
        
    def optimize_params_dict(self):
        params = list()
        params.append({'params': self.text_encoder.parameters()})
        params.append({'params': self.embedding.parameters()})
        params.append({'params': self.hiagm.parameters()})
        return params

    def forward(self, batch):
        # get distributed representation of tokens, (batch_size, max_length, embedding_dimension)
        text_embedding = self.embedding(batch['token'].to(self.config.dict['train']['device_setting']['device']))

        # get the length of sequences for dynamic rnn, (batch_size, 1)
        seq_len = batch['token_len']

        token_output = self.text_encoder(text_embedding, seq_len)

        logits = self.hiagm(token_output)

        return logits