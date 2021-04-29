import torch
import torch.nn.functional as F
from catalyst.contrib import registry
from torch import nn

import numpy as np



def IoUOneExample(pred, target):
    
    pred_area = abs(pred[2] - pred[0])*abs(pred[3] - pred[1])

    target_area = (target[2] - target[0])*(target[3] - target[1])
    

    x_p1 = min(pred[2], pred[0])
    x_p2 = max(pred[2], pred[0])

    y_p1 = min(pred[1], pred[3])
    y_p2 = max(pred[1], pred[3])


    x1 = max(x_p1, target[0])
    x2 = min(x_p2, target[2])

    y1 = max(y_p1, target[1])
    y2 = min(y_p2, target[3])
    
    overlap_area = (x2 - x1) * (y2 - y1)
    if overlap_area < 0:
        overlap_area = 0
        return 0.0

    return (overlap_area / (target_area + pred_area - overlap_area))

def IoULoss(pred, target):
    iou = IoUOneExample(pred, target)
    return 1 - iou

def GeneralizedIoUExample(pred, target):
    pred_area = abs(pred[2] - pred[0])*abs(pred[3] - pred[1])

    target_area = (target[2] - target[0])*(target[3] - target[1])
    

    x_p1 = min(pred[2], pred[0])
    x_p2 = max(pred[2], pred[0])

    y_p1 = min(pred[1], pred[3])
    y_p2 = max(pred[1], pred[3])


    x1 = max(x_p1, target[0])
    x2 = min(x_p2, target[2])

    y1 = max(y_p1, target[1])
    y2 = min(y_p2, target[3])
    
    overlap_area = (x2 - x1) * (y2 - y1)
    if overlap_area < 0:
        overlap_area = 0
        
    x1_c = min(x_p1, target[0])
    x2_c = max(x_p2, target[2])
    
    y1_c = min(y_p1, target[1])
    y2_c = max(y_p2, target[3])
    
    enclose_area = (x2_c - x1_c) * (y2_c - y1_c)
    
    return (overlap_area / (target_area + pred_area - overlap_area)) - ((enclose_area - (target_area + pred_area - overlap_area)) / enclose_area)


def GeneralizedIoULoss(pred, target):
    gen_iou = GeneralizedIoUExample(pred, target)
    return 1 - gen_iou


def EfficientIoUExample(pred, target):
    pred_area = abs(pred[2] - pred[0])*abs(pred[3] - pred[1])

    target_area = (target[2] - target[0])*(target[3] - target[1])
    

    x_p1 = min(pred[2], pred[0])
    x_p2 = max(pred[2], pred[0])

    y_p1 = min(pred[1], pred[3])
    y_p2 = max(pred[1], pred[3])


    x1 = max(x_p1, target[0])
    x2 = min(x_p2, target[2])

    y1 = max(y_p1, target[1])
    y2 = min(y_p2, target[3])
    
    overlap_area = (x2 - x1) * (y2 - y1)
    if overlap_area < 0:
        overlap_area = 0
        
    x1_c = min(x_p1, target[0])
    x2_c = max(x_p2, target[2])
    
    y1_c = min(y_p1, target[1])
    y2_c = max(y_p2, target[3])
    
    c_w = x2_c - x1_c
    w = x_p2 - x_p1
    w_t = target[2] - target[0]
    
    c_h = y2_c - y1_c
    h = y_p2 - y_p1
    h_t = target[3] - target[1]
    
    iou = (overlap_area / (target_area + pred_area - overlap_area))

    center_pred_x = (x_p2 + x_p1)/2.0
    center_pred_y = (y_p2 + y_p1)/2.0
    
    center_target_x = (target[2] + target[0])/2.0
    center_target_y = (target[3] + target[1])/2.0
    
    diag_enclosing_box = c_w**2 + c_h**2
    
    center_part = ((center_pred_x - center_target_x)**2 + (center_pred_y - center_target_y)**2) / diag_enclosing_box
    
    weight_part = ((w - w_t)**2) / (c_w**2)
    
    height_part = ((h - h_t)**2) / (h_t**2)
    
    return iou - (center_part + weight_part + height_part)

def EfficientIoULoss(pred, target):
    eff_iou = EfficientIoUExample(pred, target)
    return 1 - eff_iou

def FocalEfficientIoULoss(pred, target, gamma = 0.5):
    iou = IoUOneExample(pred, target)
    eiou_loss = EfficientIoULoss(pred, target)
    return (iou**gamma)*eiou_loss
    
@registry.Criterion
class IoU(nn.Module):
    def __init__(self, bce_coeff = 0.2):
        super(IoU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.bce_coeff = bce_coeff
     
    def forward(self, preds, target):
        n = len(target)
        loss_sum = 0
        for i in range(n):
            loss_sum += IoULoss(self.sigmoid(preds[i][1:]), target[i][1:])
            
        bce = F.binary_cross_entropy_with_logits(preds[:, 0], target[:, 0])
        return self.bce_coeff * bce + (loss_sum / n)

@registry.Criterion
class GeneralizedIoU(nn.Module):
    def __init__(self, bce_coeff = 0.2):
        super(GeneralizedIoU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.bce_coeff = bce_coeff
     
    def forward(self, preds, target):
        n = len(target)
        loss_sum = 0
        for i in range(n):
            loss_sum += GeneralizedIoULoss(self.sigmoid(preds[i][1:]), target[i][1:])
            
        bce = F.binary_cross_entropy_with_logits(preds[:, 0], target[:, 0])
        return self.bce_coeff * bce + (loss_sum / n)
    
    
@registry.Criterion
class EfficientIoU(nn.Module):
    def __init__(self, bce_coeff = 0.2):
        super(EfficientIoU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.bce_coeff = bce_coeff
     
    def forward(self, preds, target):
        n = len(target)
        loss_sum = 0
        for i in range(n):
            loss_sum += EfficientIoULoss(self.sigmoid(preds[i][1:]), target[i][1:])
            
        bce = F.binary_cross_entropy_with_logits(preds[:, 0], target[:, 0])
        return self.bce_coeff * bce + (loss_sum / n)
    
@registry.Criterion
class FocalEfficientIoU(nn.Module):
    def __init__(self, bce_coeff = 0.2, gamma = 0.5):
        super(FocalEfficientIoU, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.bce_coeff = bce_coeff
        self.gamma = gamma
        
    def forward(self, preds, target):
        n = len(target)
        loss_sum = 0
        for i in range(n):
            loss_sum += FocalEfficientIoULoss(self.sigmoid(preds[i][1:]), target[i][1:], self.gamma)
            
        bce = F.binary_cross_entropy_with_logits(preds[:, 0], target[:, 0])
        return self.bce_coeff * bce + (loss_sum / n)



def get_loss(name_loss, bce_coeff, gamma = 0.5):
    LEVELS = {
        'IoU': IoU(bce_coeff),
        'GeneralizedIoU': GeneralizedIoU(bce_coeff),
        'EfficientIoU': EfficientIoU(bce_coeff),
        'FocalEfficientIoU': FocalEfficientIoU(bce_coeff, gamma)
    }
    return LEVELS[name_loss]