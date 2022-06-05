import torch.nn as nn

import torch
import time

class My_NER(nn.Module):
    def __init__(self):
        super(My_NER, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=768,nhead=12) 
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=12)
        
        self.dropout = nn.Dropout(p = 0.1)
        self.fc = nn.Linear(in_features=768, out_features=9, bias=True)
        
        
    def forward(self, x):
        out = self.transformer_encoder(x)
        out = self.dropout(x)
        out = self.fc(x)
        return out


#model = My_NER()
#src = torch.rand(4, 32, 768)
'''
model = model.to('cuda:0')

src = src.to('cuda:0')

with torch.no_grad():
    out = model(src)
    torch.cuda.synchronize()

#op_time = nn.modules.module.op_time
#print(len(op_time))
#print(nn.modules.module.op_name)
'''