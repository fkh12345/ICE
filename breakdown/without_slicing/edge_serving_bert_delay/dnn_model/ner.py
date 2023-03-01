import torch.nn as nn

import torch
import time

class My_NER(nn.Module):
    def __init__(self):
        super(My_NER, self).__init__()
        self.size = torch.rand(1, 32, 768)
        encoder_layer = nn.TransformerEncoderLayer(d_model=768,nhead=12) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=12)
        
        self.dropout = nn.Dropout(p = 0.1)
        self.fc = nn.Linear(in_features=768, out_features=9, bias=True)
        
        
    def forward(self, x):
        out = self.transformer_encoder(x)
        out = self.dropout(x)
        out = self.fc(x)
        return out


