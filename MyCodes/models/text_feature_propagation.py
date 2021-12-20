import torch
import torch.nn as nn

class HiAGM_TP(nn.Module):
    def __init__(self, config, label_map, graph_model, device):
        super(HiAGM_TP, self).__init__()
        self.config = config
        self.label_map = label_map
        self.graph_model = graph_model
        self.device = device

        self.transformation = nn.Linear(self.config.dict['model']['linear_transformation']['text_dimension'],
                                         len(self.label_map) * self.config.dict['model']['linear_transformation']['text_dimension'])
        self.linear = nn.Linear(len(self.label_map) * self.config.dict['embedding']['label']['dimension'], len(self.label_map))

        self.transformation_dropout = nn.Dropout(p=self.config.dict['model']['linear_transformation']['dropout'])
        self.dropout = nn.Dropout(p=self.config.dict['model']['classifier']['dropout'])
    
    def forward(self, text_feature):
        text_feature = torch.cat(text_feature, dim=1)
        text_feature = text_feature.view(text_feature.shape[0], -1)

        text_feature = self.transformation_dropout(self.transformation(text_feature))
        text_feature = text_feature.view(text_feature.shape[0],
                                         len(self.label_map),
                                         self.config.dict['model']['linear_transformation']['node_dimension'])

        label_wise_text_feature = self.graph_model(text_feature)
        print(label_wise_text_feature.size())
        logits = self.dropout(self.linear(label_wise_text_feature.view(label_wise_text_feature.shape[0], -1)))
        return logits