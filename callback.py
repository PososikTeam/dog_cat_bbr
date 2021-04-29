from sklearn.metrics import confusion_matrix, f1_score, fbeta_score, cohen_kappa_score
from catalyst.dl import AccuracyCallback, MetricCallback, State, Callback, CriterionCallback, OptimizerCallback
from pytorch_toolbelt.utils.torch_utils import to_numpy
from catalyst.core.callback import Callback, CallbackOrder
from scipy.special import softmax

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from loss import IoUOneExample
import math

# custom function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# define vectorized sigmoid
sigmoid_v = np.vectorize(sigmoid)



class MyAccuracyCallback(Callback):
    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "accuracy01"):
        """
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        super().__init__(CallbackOrder.metric)
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.sum_correct = 0
        self.count = 0

    def on_loader_start(self, state):
        self.sum_correct = 0
        self.count = 0

    def on_batch_end(self, state):

        targets = state.input[self.input_key].detach()
        outputs = state.output[self.output_key].detach()
        
        targets = np.array(targets[:, 0].cpu())
        outputs = outputs[:, 0]
        
        outputs = np.array([1 if a > 0.5 else 0 for a in outputs])
        
        self.sum_correct += int((targets  == outputs).sum())
        self.count += targets.shape[0]


    def on_loader_end(self, state: State):
        state.loader_metrics[self.prefix] = self.sum_correct / self.count



def IoUAll(pred, target):
    n = len(pred)
    iou = 0
    for i in range(n):
        iou += IoUOneExample(sigmoid_v(pred[i]), target[i])
    
    return iou / n

class IoUCallback(Callback):
    def __init__(self,
                 input_key: str = "targets",
                 output_key: str = "logits",
                 prefix: str = "IoU"):
        """
        :param input_key: input key to use for precision calculation; specifies our `y_true`.
        :param output_key: output key to use for precision calculation; specifies our `y_pred`.
        """
        super().__init__(CallbackOrder.metric)
        self.prefix = prefix
        self.output_key = output_key
        self.input_key = input_key
        self.targets = []
        self.predictions = []

    def on_loader_start(self, state):
        self.targets = []
        self.predictions = []

    def on_batch_end(self, state):

        targets = to_numpy(state.input[self.input_key].detach())
        outputs = to_numpy(state.output[self.output_key].detach())
        

        self.targets.extend(targets[:, 1:])
        self.predictions.extend(outputs[:, 1:])

    def on_loader_end(self, state: State):
        # predictions = to_numpy(self.predictions)
        # predictions = np.argmax(predictions, axis=1)     
        score = IoUAll(self.predictions, self.targets)
        state.loader_metrics[self.prefix] = float(score)