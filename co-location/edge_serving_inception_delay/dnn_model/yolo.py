import numpy as np
import torch
import torch.nn as nn
from time import time

class Cat(nn.Module):
    def __init__(self):
        super(Cat, self).__init__()
    def forward(self, x):
        out = torch.cat(x, dim=1)
        return out

class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
    def forward(self, x, y):
        out = x + y
        return out
class DarknetLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bnorm = True, leaky = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias = False if bnorm else True)
        self.bnorm = nn.BatchNorm2d(out_channels, eps = 1e-3) if bnorm else None
        self.leaky = nn.LeakyReLU(0.1) if leaky else None
    def forward(self, x):
        x = self.conv(x)
        if self.bnorm is not None:
            x = self.bnorm(x)
        if self.leaky is not None:
            x = self.leaky(x)
        return x

class DarknetBlock(nn.Module):
    def __init__(self, layers, skip = False):
        super().__init__()
        self.skip = skip
        self.layers = nn.ModuleDict()
        for i in range(len(layers)):
            self.layers[layers[i]['id']] = DarknetLayer(layers[i]['in_channels'], layers[i]['out_channels'], layers[i]['kernel_size'],layers[i]['stride'], layers[i]['padding'], layers[i]['bnorm'], layers[i]['leaky'])
        self.add = Add()
    def forward(self, x):
        count = 0
        for _, layer in self.layers.items():
            if count == (len(self.layers) - 2) and self.skip:
                skip_connection = x
            count += 1
            x = layer(x)
        out = x
        if(self.skip):
            out = self.add(x, skip_connection)
        else:
            out = x
        return out


class Yolov3(nn.Module):
    def __init__(self):
        super(Yolov3, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners = False)
        #layer0 -> layer4, input = (3, 416, 416), flow_out = (64, 208, 208)
        self.blocks = nn.ModuleDict()
        self.blocks['block0_4'] = DarknetBlock([
            {'id': 'layer_0', 'in_channels': 3, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_1', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_2', 'in_channels': 64, 'out_channels': 32, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_3', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer5 -> layer8, input = (64, 208, 208), flow_out = (128, 104, 104)
        self.blocks['block5_8'] = DarknetBlock([
            {'id': 'layer_5', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_6', 'in_channels': 128, 'out_channels': 64, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_7', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer9 -> layer11, input = (128, 104, 104), flow_out = (128, 104, 104)
        self.blocks['block9_11'] = DarknetBlock([
            {'id': 'layer_9', 'in_channels': 128, 'out_channels': 64, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_10', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer12 -> layer15, input = (128, 104, 104), flow_out = (256, 52, 52)
        self.blocks['block12_15'] = DarknetBlock([
            {'id': 'layer_12', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_13', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_14', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer16 -> layer36, input = (256, 52, 52), flow_out = (256, 52, 52)
        self.blocks['block16_18'] = DarknetBlock([
            {'id': 'layer_16', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_17', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block19_21'] = DarknetBlock([
            {'id': 'layer_19', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_20', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block22_24'] = DarknetBlock([
            {'id': 'layer_22', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_23', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block25_27'] = DarknetBlock([
            {'id': 'layer_25', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_26', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block28_30'] = DarknetBlock([
            {'id': 'layer_28', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_29', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block31_33'] = DarknetBlock([
            {'id': 'layer_31', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_32', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block34_36'] = DarknetBlock([
            {'id': 'layer_34', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_35', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer37 -> layer40, input = (256, 52, 52), flow_out = (512, 26, 26)
        self.blocks['block37_40'] = DarknetBlock([
            {'id': 'layer_37', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 2, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_38', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_39', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer41 -> layer61, input = (512, 26, 26), flow_out = (512, 26, 26)
        self.blocks['block41_43'] = DarknetBlock([
            {'id': 'layer_41', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_42', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block44_46'] = DarknetBlock([
            {'id': 'layer_44', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_45', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block47_49'] = DarknetBlock([
            {'id': 'layer_47', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_48', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block50_52'] = DarknetBlock([
            {'id': 'layer_50', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_51', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block53_55'] = DarknetBlock([
            {'id': 'layer_53', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_54', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block56_58'] = DarknetBlock([
            {'id': 'layer_56', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_57', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block59_61'] = DarknetBlock([
            {'id': 'layer_59', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_60', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer62 -> layer65, input = (512, 26, 26), flow_out = (1024, 13, 13)
        self.blocks['block62_65'] = DarknetBlock([
            {'id': 'layer_62', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 2, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_63', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_64', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer66 -> layer74, input = (1024, 13, 13), flow_out = (1024, 13, 13)
        self.blocks['block66_68'] = DarknetBlock([
            {'id': 'layer_66', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_67', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block69_71'] = DarknetBlock([
            {'id': 'layer_69', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_70', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        self.blocks['block72_74'] = DarknetBlock([
            {'id': 'layer_72', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_73', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True}
        ])
        #layer75 -> layer79, input = (1024, 13, 13), flow_out = (512, 13, 13)
        self.blocks['block75_79'] = DarknetBlock([
            {'id': 'layer_75', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_76', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_77', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_78', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_79', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True}
        ], skip = False)
        #layer80 -> layer82, input = (512, 13, 13), yolo_out = (255, 13, 13)
        self.blocks['yolo_82'] = DarknetBlock([
            {'id': 'layer_80', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_81', 'in_channels': 1024, 'out_channels': 255, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': False, 'leaky': False}
        ], skip = False)
        #layer83 -> layer86, input = (512, 13, 13), -> (256, 13, 13) -> upsample and concate layer61(512, 26, 26), flow_out = (768, 26, 26)
        self.blocks['block83_86'] = DarknetBlock([
            {'id': 'layer_84', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True}
        ], skip = False)
        #layer87 -> layer91, input = (768, 26, 26), flow_out = (256, 26, 26)
        self.blocks['block87_91'] = DarknetBlock([
            {'id': 'layer_87', 'in_channels': 768, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_88', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_89', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_90', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_91', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True}
        ], skip = False)
        #layer92 -> layer94, input = (256, 26, 26), yolo_out = (255, 26, 26)
        self.blocks['yolo_94'] = DarknetBlock([
            {'id': 'layer_92', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_93', 'in_channels': 512, 'out_channels': 255, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': False, 'leaky': False}
        ], skip = False)
        #layer95 -> layer98, input = (256, 26, 26), -> (128, 26, 26) -> upsample and concate layer36(256, 52, 52), flow_out = (384, 52, 52)
        self.blocks['block95_98'] = DarknetBlock([
            {'id': 'layer_96', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True}
        ], skip = False)
        #layer99 -> layer106, input = (384, 52, 52), yolo_out = (255, 52, 52)
        self.blocks['yolo_106'] = DarknetBlock([
            {'id': 'layer_99', 'in_channels': 384, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_100', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_101', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_102', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_103', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': True, 'leaky': True},
            {'id': 'layer_104', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding' : 1, 'bnorm': True, 'leaky': True},
            {'id': 'layer_105', 'in_channels': 256, 'out_channels': 255, 'kernel_size': 1, 'stride': 1, 'padding' : 0, 'bnorm': False, 'leaky': False}
        ], skip = False)
        self.cat = Cat()

    def forward(self, x):
        x = self.blocks['block0_4'](x)
        x = self.blocks['block5_8'](x)
        x = self.blocks['block9_11'](x)
        x = self.blocks['block12_15'](x)
        x = self.blocks['block16_18'](x)
        x = self.blocks['block19_21'](x)
        x = self.blocks['block22_24'](x)
        x = self.blocks['block25_27'](x)
        x = self.blocks['block28_30'](x)
        x = self.blocks['block31_33'](x)
        x = self.blocks['block34_36'](x)
        skip36 = x
        x = self.blocks['block37_40'](x)
        x = self.blocks['block41_43'](x)
        x = self.blocks['block44_46'](x)
        x = self.blocks['block47_49'](x)
        x = self.blocks['block50_52'](x)
        x = self.blocks['block53_55'](x)
        x = self.blocks['block56_58'](x)
        x = self.blocks['block59_61'](x)
        skip61 = x
        x = self.blocks['block62_65'](x)
        x = self.blocks['block66_68'](x)
        x = self.blocks['block69_71'](x)
        x = self.blocks['block72_74'](x)
        x = self.blocks['block75_79'](x)
        x = self.blocks['block83_86'](x)
        x = self.upsample(x)
        '''
        x = self.cat((x, skip61))
        x = self.blocks['block87_91'](x)
        x = self.blocks['block95_98'](x)
        x = self.upsample(x)
        x = self.cat((x, skip36))
        yolo_106 = self.blocks['yolo_106'](x)
        '''
        return x
        #return yolo_106      