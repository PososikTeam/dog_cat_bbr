import math
import os
from typing import Tuple, List

import albumentations as A
import cv2
import numpy as np
import pandas as pd
# from pytorch_toolbelt.utils import fs
# from pytorch_toolbelt.utils.fs import id_from_fname
# from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils import compute_sample_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from augmentation import get_train_transform, get_test_transform
import matplotlib.pyplot as plt

def crop_img(img, points):
    width = img.shape[1]
    height = img.shape[0]

    points = [int(width * points[0]), int(height * points[1]), int(width * points[2]), int(height * points[3])]

    if points[0] != 0:
        crop_x1 = np.random.choice(range(0, points[0]))
    else:
        crop_x1 = 0
    
    if points[1] != 0:
        crop_y1 = np.random.choice(range(0, points[1]))
    else:
        crop_y1 = 0

    if points[2] != width:
        crop_x2 = np.random.choice(range(points[2], width))
    else:
        crop_x2 = width

    if points[3] != height:
        crop_y2 = np.random.choice(range(points[3], height))
    else:
        crop_y2 = height

    new_points = [points[0] - crop_x1, points[1] - crop_y1, points[2] - crop_x1, points[3] - crop_y1]

    croped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]

    crop_width = croped_img.shape[1]
    crop_height = croped_img.shape[0]

    float_points = [new_points[0]/crop_width, new_points[1]/crop_height ,new_points[2]/crop_width, new_points[3]/crop_height]
    
    return croped_img, float_points


class TaskDataset(Dataset):
    def __init__(self, images, label_targets, box_targets, transform: A.Compose, crop_prob = 0.25):
        

        self.images = images
        self.box_targets = box_targets
        self.label_targets = label_targets
        self.transform = transform
        self.crop_prob = crop_prob

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = cv2.imread(self.images[item])  # Read with OpenCV instead PIL. It's faster
        if image is None:
            raise FileNotFoundError(self.images[item])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = self.box_targets[item]
        label = self.label_targets[item]
        
        if np.random.random() <= self.crop_prob:
            # print(bboxes)
            image, bboxes = crop_img(image, bboxes)
            
        if self.transform == None:
            answ = {'image': image, 'targets': [label, bboxes[0], bboxes[1], bboxes[2], bboxes[3]]}
        else:
            data = self.transform(image=image, bboxes=[bboxes], category_ids=[label])

            xmin, ymin, xmax, ymax = data['bboxes'][0]

            category_ids = data['category_ids'][0]

            target = np.array([category_ids, xmin, ymin, xmax, ymax])

            answ = {'image': data['image'], 'targets': target}
        return answ


def split_train_valid(img_names, boxes, labels, fold=None, folds=4, random_state=42):
    """
    Common train/test split function
    :param x:
    :param y:
    :param fold:
    :param folds:
    :param random_state:
    :return:
    """
    train_x, train_box, train_y = [], [], []
    valid_x, valid_box, valid_y = [], [], []

    if fold is not None:
        assert 0 <= fold < folds
        skf = StratifiedKFold(n_splits=folds, random_state=random_state, shuffle=True)
        

        for fold_index, (train_index, test_index) in enumerate(skf.split(img_names, labels)):
            if fold_index == fold:
                train_x = img_names[train_index]
                train_box = boxes[train_index]
                train_y = labels[train_index]

                
                valid_x = img_names[test_index]
                valid_box = boxes[test_index]
                valid_y = labels[test_index]
                break
    

    assert len(train_x) and len(train_y) and len(valid_x) and len(valid_y)
    assert len(train_x) == len(train_y)
    assert len(valid_x) == len(valid_y)
    return train_x, valid_x, train_y, valid_y, train_box, valid_box


def get_current_train(data_dir, csv_name_file):

    df = pd.read_csv(csv_name_file)
    x = np.array(df['name'].apply(lambda x: os.path.join(data_dir,  f'{x}')))
    box = np.array([df['xmin'], df['ymin'], df['xmax'], df['ymax']]).T
    y = np.array(df['id']).T

    return x, y, box


def get_datasets_universal(data_dir='data',
        csv_train_file_name = 'train.csv',
        csv_test_file_name = 'test.csv',
        image_size=(512, 512),
        augmentation='medium',
        random_state=42):
    
        train_x, train_y, train_box = get_current_train(data_dir, csv_train_file_name)
        valid_x, valid_y, valid_box = get_current_train(data_dir, csv_test_file_name)
        
        train_transform = get_train_transform(augmentation = augmentation, image_size = image_size)
        valid_transform = get_test_transform(image_size = image_size)

        train_ds = TaskDataset(train_x, train_y, train_box, transform=train_transform)
        valid_ds = TaskDataset(valid_x, valid_y, valid_box, transform=valid_transform)
        
        return train_ds, valid_ds


def get_datasets(
        data_dir='data',
        csv_file_name = 'train.csv',
        image_size=(512, 512),
        augmentation='medium',
        random_state=42,
        fold=None,
        folds=4):


    trainset_sizes = []
    data_split = [], [], [], [], [], []

    x, y, box = get_current_train(data_dir, csv_file_name)

    
    #split = train_x, valid_x, train_y, valid_y, train_box, valid_box
    split = split_train_valid(x, box, y, fold=fold, folds=folds,  random_state=random_state)

    train_x, valid_x, train_y, valid_y, train_box, valid_box = split



    train_transform = get_train_transform(augmentation = augmentation, image_size = image_size)
    valid_transform = get_test_transform(image_size = image_size)

    train_ds = TaskDataset(train_x, train_y, train_box, transform=train_transform)

    valid_ds = TaskDataset(valid_x, valid_y, valid_box, transform=valid_transform)

    return train_ds, valid_ds


def get_dataloaders(train_ds, valid_ds,
                    batch_size,
                    num_workers = 1):
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, pin_memory=True, num_workers = num_workers, shuffle = True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, pin_memory=True, num_workers = num_workers, shuffle = True)

    return train_dl, valid_dl