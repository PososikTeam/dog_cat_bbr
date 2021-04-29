import albumentations as A
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset, DataLoader
import cv2

BORDER_CONSTANT = 0
BORDER_REFLECT = 2




    



def get_medium_augmentations(image_size):
    return A.Compose([
        A.OneOf([
                A.Transpose(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.ShiftScaleRotate(),
                A.NoOp()
            ], p = 0.5),
            

            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
                A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1)),
                A.NoOp()
            ], p = 0.5),
            
            A.Cutout(p=0.15),

            A.Resize(image_size[0], image_size[1], p = 1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p = 1.0),
            ToTensor()
        ], 
            bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids']),
        )

def get_hard_augmentations(image_size):
    return A.Compose([ 

            A.OneOf([A.CLAHE(clip_limit=(10, 10), tile_grid_size=(3, 3)),
                A.FancyPCA(alpha=0.4),
                A.NoOp(),
                ], p = 0.5),

            A.OneOf([
                A.Transpose(),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.ShiftScaleRotate(),
                A.NoOp()
            ], p = 0.5),
            

            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
                A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1)),
                A.RandomGamma(gamma_limit=(50, 150)),
                A.NoOp()
            ], p = 0.5),
            
            A.Cutout(p=0.15),

            A.Resize(image_size[0], image_size[1], p = 1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p = 1.0),
            ToTensor()
        ], 
            bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids']),
        )




def light_aug(image_size):
    return A.Compose(
        [A.HorizontalFlip(p = 0.5), 
        A.VerticalFlip(p = 0.5),
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p = 1.0),
        ToTensor()
        ],
        bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids']),
    )
   


def get_test_transform(image_size):
    return A.Compose([A.Resize(image_size[0], image_size[1], p = 1.0), 
                      A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p = 1.0), 
                      ToTensor()], 
                    bbox_params=A.BboxParams(format='albumentations', label_fields=['category_ids']))


def get_train_transform(augmentation, image_size):
    if augmentation == 'light':
        return light_aug(image_size)
    if augmentation == 'medium':
        return get_medium_augmentations(image_size)
    if augmentation == 'hard':
        return get_hard_augmentations(image_size)
   
    
    
