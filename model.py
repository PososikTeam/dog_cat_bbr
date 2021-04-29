from typing import Callable, List, Tuple 

import os
import torch
import catalyst

from catalyst.dl import utils


SEED = 42
utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)


from torch import nn
import torch.nn.functional as F
import pretrainedmodels


from efficientnet_pytorch import EfficientNet
import timm


"""
EfficientNetB0 - (224, 224, 3)
EfficientNetB1 - (240, 240, 3)
EfficientNetB2 - (260, 260, 3)
EfficientNetB3 - (300, 300, 3)
EfficientNetB4 - (380, 380, 3)
EfficientNetB5 - (456, 456, 3)
EfficientNetB6 - (528, 528, 3)
EfficientNetB7 - (600, 600, 3)
"""
# 'mobilenetv2_100',
#  'mobilenetv2_110d',
#  'mobilenetv2_120d',
#  'mobilenetv2_140',
#  'mobilenetv3_large_100',
#  'mobilenetv3_rw',



class mobilenet(torch.nn.Module):
    def __init__(self, name, output_layer = 5):
        super(mobilenet, self).__init__()
        self.model = timm.create_model(name, pretrained=True)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, output_layer)
        
        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        y_pred = self.model(x)
        return y_pred


    

def get_model(model_name, output_layer = 5):
    return mobilenet(model_name, output_layer=output_layer)
    