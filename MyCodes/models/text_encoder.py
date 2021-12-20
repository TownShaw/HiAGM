import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self,
                 layers,
                 in_dim,
                 out_dim,
                 bias=True,
                 batch_first=True,
                 dropout=0.0,
                 bidirectional=True):
        super(GRU, self).__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_layers = layers
        self.gru = nn.GRU(input_size=in_dim,
                          hidden_size=out_dim,
                          num_layers=layers,
                          bias=bias,
                          batch_first=batch_first,
                          dropout=dropout,
                          bidirectional=bidirectional)
    
    def forward(self, inputs, seq_len=None, init_state=None, ori_state=False):
        if seq_len is not None:
            seq_len = seq_len.int()
            sorted_seq_len, indices = torch.sort(seq_len, descending=True)
            if self.batch_first:
                sorted_inputs = inputs[indices]
            else:
                sorted_inputs = inputs[:, indices]
            packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(sorted_inputs, sorted_seq_len, batch_first=self.batch_first)

            outputs, states = self.gru(packed_inputs, init_state)

            if self.bidirectional:
                last_layer_state = states[2 * (self.num_layers - 1):]
                last_layer_state = torch.cat((last_layer_state[0], last_layer_state[1]), dim=1)
            else:
                last_layer_state = states[self.num_layers - 1]
                last_layer_state = last_layer_state[0]

            _, reversed_indices = torch.sort(indices, descending=False)
            last_layer_state = last_layer_state[reversed_indices]
            padding_out, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)

            if self.batch_first:
                padding_out = padding_out[reversed_indices]
            else:
                padding_out = padding_out[:, reversed_indices]
            return padding_out, last_layer_state


class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        self.config = config
        self.rnn = GRU(layers=config.dict['text_encoder']['RNN']['num_layers'],
                       in_dim=config.dict['embedding']['token']['dimension'],
                       out_dim=config.dict['text_encoder']['RNN']['hidden_dimension'],
                       batch_first=True,
                       dropout=config.dict['text_encoder']['RNN']['dropout'],
                       bidirectional=config.dict['text_encoder']['RNN']['bidirectional'])
        hidden_dimension = config.dict['text_encoder']['RNN']['hidden_dimension']
        if config.dict['text_encoder']['RNN']['bidirectional']:
            hidden_dimension *= 2
        self.conv_list = nn.ModuleList()
        self.kernel_size = config.dict['text_encoder']['CNN']['kernel_size']
        for kernel_size in self.kernel_size:
            self.conv_list.append(nn.Conv1d(in_channels=hidden_dimension,
                                            out_channels=config.dict['text_encoder']['CNN']['num_kernel'],
                                            kernel_size=kernel_size,
                                            padding=kernel_size // 2))
        self.top_k = config.dict['text_encoder']['topK_max_pooling']
        self.rnn_dropout = torch.nn.Dropout(p=config.dict['text_encoder']['RNN']['dropout'])
    
    def forward(self, inputs, seq_lens):
        # print(inputs.size(), seq_lens.size())
        outputs, _ = self.rnn(inputs, seq_lens)
        outputs = self.rnn_dropout(outputs)
        outputs = outputs.transpose(1, 2)
        # print(outputs.size())
        topk_conv_list = []
        for idx, conv in enumerate(self.conv_list):
            conv_output = F.relu(conv(outputs))
            topk_conv_output = torch.topk(conv_output, self.config.dict['text_encoder']['topK_max_pooling'])[0].view(outputs.size(0), -1)
            topk_conv_output = topk_conv_output.unsqueeze(1)
            topk_conv_list.append(topk_conv_output)
        return topk_conv_list