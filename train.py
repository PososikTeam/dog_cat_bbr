import argparse
import collections
import json
import multiprocessing
import os
import numpy as np
from datetime import datetime

import torch
import catalyst
from catalyst.dl import SupervisedRunner, EarlyStoppingCallback
from catalyst.utils import load_checkpoint, unpack_checkpoint

from pytorch_toolbelt.utils.random import set_manual_seed, get_random_name
from dataset import get_datasets, get_dataloaders, get_datasets_universal
from model import get_model
from loss import get_loss
from callback import MyAccuracyCallback, IoUCallback
from optimizer import get_optim
from catalyst.dl import AccuracyCallback, OptimizerCallback, CheckpointCallback
from torch import nn
from torch.optim import lr_scheduler
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('-v', '--verbose', action='store_true')

    parser.add_argument('-dd', '--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('-train_csv', '--train_csv', type=str, default='train.csv')
    parser.add_argument('-test_csv', '--test_csv', type=str, default='test.csv')

    parser.add_argument('-optim', '--optimizer', type = str, default = 'adam')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('-weight_decay', '--weight_decay', type = float, default = 1e-6)
    parser.add_argument('-scheduler', '--scheduler', type = str, default='CosineAnnealingWarmRestarts')

    
    
    
    
    parser.add_argument('-m', '--model', type=str, default='mobilenetv2_120d', help='')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='Batch Size during training, e.g. -b 64')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Epoch to run')
    parser.add_argument('-s', '--sizes', default=380, type=int, help='Image size for training & inference')
    parser.add_argument('-a', '--augmentations', default='medium', type=str, help='')


    parser.add_argument('-loss', '--loss', type = str, default = 'EfficientIoU')
    parser.add_argument('-metric', '--metric', type = str, default = 'IoU')
    parser.add_argument('-bce_coeff', '--bce_coeff', type = float, default = 0.2)
    


    args = parser.parse_args()

    seed = args.seed
    verbose = args.verbose

    data_dir = args.data_dir
    train_csv = args.train_csv
    test_csv = args.test_csv

    optim_name = args.optimizer
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    scheduler_name = args.scheduler

    model_name = args.model
    batch_size = args.batch_size
    epochs = args.epochs
    size = args.sizes
    augmentation_name = args.augmentations

    main_metric = args.metric
    loss_name = args.loss
    bce_coeff = args.bce_coeff

  
    

    current_time = datetime.now().strftime('%b%d_%H_%M')
    random_name = get_random_name()


    torch.cuda.empty_cache()
    checkpoint_prefix = f'{model_name}_{size}_{augmentation_name}'

    
    directory_prefix = f'{current_time}_{checkpoint_prefix}_{optim_name}_{learning_rate}_{loss_name}'
    log_dir = os.path.join('runs', directory_prefix)
    os.makedirs(log_dir, exist_ok=False)


    set_manual_seed(seed)

    model = get_model(model_name)
    model = model.cuda()

  
    image_size = (size, size)

  

    train_ds, valid_ds = get_datasets_universal(data_dir=data_dir,
                                    csv_train_file_name = train_csv,
                                    csv_test_file_name = test_csv,
                                    image_size=image_size,
                                    augmentation=augmentation_name)

    train_loader, valid_loader = get_dataloaders(train_ds, valid_ds,
                                                    batch_size=batch_size,
                                                    num_workers = 2)


    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    runner = SupervisedRunner(input_key='image')

    
    criterions = get_loss(loss_name, bce_coeff)
    optimizer = get_optim(optim_name, model, learning_rate, weight_decay)

    min_lr = 1e-6
    if scheduler_name == 'CosineAnnealingLR':
        used_scheduler = lr_scheduler.CosineAnnealingLR(optimizer,  T_max=epochs, eta_min=min_lr)
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        used_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, T_mult=1, eta_min=min_lr)

        
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    callbacks = [MyAccuracyCallback(), IoUCallback()]


    runner.train(
        fp16=dict(amp=True),
        model=model,
        verbose = verbose,
        criterion=criterions,
        optimizer=optimizer,
        scheduler= scheduler,
        callbacks=callbacks,
        num_epochs=epochs,
        loaders=loaders,
        main_metric=main_metric,
        logdir=log_dir,
        minimize_metric=False,
    )


if __name__ == '__main__':
    main()