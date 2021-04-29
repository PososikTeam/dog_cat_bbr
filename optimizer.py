import torch
import catalyst

  
import torch
from torch.optim.optimizer import Optimizer



def get_optim(optim_name, model, lr, weight_decay):
    d = {
        'adam': torch.optim.Adam(model.parameters(), weight_decay = weight_decay, lr = lr, amsgrad=False),
        'sgd': torch.optim.SGD(model.parameters(), lr, momentum=0.9),
        'radam': catalyst.contrib.nn.optimizers.radam.RAdam(model.parameters(), lr),
    }
    return d[optim_name]
    