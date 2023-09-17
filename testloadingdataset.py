import copy
import warnings 
import os 

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset

# from .dataset import ImageDataset 
from dataset import ImageDataset 

image_size = 224 
extra_train_transforms = None 

root = '/rscratch/data/imagenet12/imagenet' 

__all__ = ['ImageNet'] 

train_transforms_pre = [
    transforms.RandomResizedCrop(image_size),
    transforms.RandomHorizontalFlip()
] 

train_transforms_post = [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
] 

if extra_train_transforms is not None:
    if not isinstance(extra_train_transforms, list):
        extra_train_transforms = [extra_train_transforms]
    for ett in extra_train_transforms:
        if isinstance(ett, (transforms.LinearTransformation, transforms.Normalize, transforms.RandomErasing)):
            train_transforms_post.append(ett)
        else:
            train_transforms_pre.append(ett)
train_transforms = transforms.Compose(train_transforms_pre + train_transforms_post)

test_transforms = [
    transforms.Resize(int(image_size / 0.875)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
test_transforms = transforms.Compose(test_transforms) 
print("root: ", root) 

# train = datasets.ImageNet(root=root, split='train', download=False, transform=train_transforms) 
# train = datasets.ImageFolder(root = os.path.join(root, 'train'), 
root_train = os.path.join(root, 'train') 
train = ImageDataset(root = root_train, reader = '', class_map = '', load_bytes = False) 
# train = datasets.ImageNet(os.path.join(root, 'train'), download = False, transform=train_transforms) 
root_val = os.path.join(root, 'val') 
test = ImageDataset(root = root_val, reader = '', class_map = '', load_bytes = False) 
# test = datasets.ImageNet(os.path.join(root, 'val'), download=False, transform=test_transforms) 
