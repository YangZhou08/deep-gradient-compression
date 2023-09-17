import copy
import warnings

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset

from .utils import split_train_val_indices
from ..dataset import Dataset 

from dataset import ImageDataset 
from torchvision.datasets.folder import ImageFolder 
import os 

__all__ = ['ImageNet']

# filter warnings for corrupted data
warnings.filterwarnings('ignore')


class ImageNet(Dataset):
    def __init__(self, root, num_classes, image_size, val_ratio=None, extra_train_transforms=None):
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
        # test = datasets.ImageNet(root=root, split='val', download=False, transform=test_transforms) 
        # train = datasets.ImageNet(root=root, split='train', download=False, transform=train_transforms) 
        # train = datasets.ImageFolder(root = os.path.join(root, 'train'), 
        root_train = os.path.join(root, 'train') 
        # train = ImageDataset(root = root_train, reader = '', class_map = '', load_bytes = False, transform = train_transforms) 
        train = ImageFolder(root = root_train, transform = train_transforms) 
        # train = datasets.ImageNet(os.path.join(root, 'train'), download = False, transform=train_transforms) 
        root_val = os.path.join(root, 'val') 
        # test = ImageDataset(root = root_val, reader = '', class_map = '', load_bytes = False, transform = test_transforms) 
        test = ImageFolder(root = root_val, transform = test_transforms) 
        # test = datasets.ImageNet(os.path.join(root, 'val'), download=False, transform=test_transforms) 

        # sample classes by strided indexing
        class_indices = dict()
        for k in range(num_classes):
            class_indices[k * (1000 // num_classes)] = k

        # reduce dataset to sampled classes
        for dataset in [train, test]:
            dataset.samples = [(x, class_indices[idx]) for x, idx in dataset.samples if idx in class_indices]
            dataset.targets = [class_indices[idx] for idx in dataset.targets if idx in class_indices]
            wnids, wnid_to_idx, classes, class_to_idx = [], dict(), [], dict()
            for idx, (wnid, clss) in enumerate(zip(dataset.wnids, dataset.classes)):
                if idx in class_indices:
                    wnids.append(wnid)
                    wnid_to_idx[wnid] = class_indices[idx]
                    classes.append(clss)
                    for c in clss:
                        class_to_idx[c] = class_indices[idx]
            dataset.wnids, dataset.wnid_to_idx = wnids, wnid_to_idx
            dataset.classes, dataset.class_to_idx = classes, class_to_idx

        if val_ratio is None:
            super(ImageNet, self).__init__(train=train, test=test)
        else:
            train_indices, val_indices = split_train_val_indices(
                targets=train.targets, val_ratio=val_ratio, num_classes=num_classes
            )
            val = train.__new__(type(train))
            val.__dict__ = copy.deepcopy(train.__dict__)
            val.transform = test_transforms
            train = Subset(train, indices=train_indices)
            val = Subset(val, indices=val_indices)
            super(ImageNet, self).__init__(train=train, val=val, test=test) 

class ImageNet2(Dataset): 
    def __init__(self, root, num_classes, image_size, val_ratio=None, extra_train_transforms=None): 
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
        # test = datasets.ImageNet(root=root, split='val', download=False, transform=test_transforms) 
        # train = datasets.ImageNet(root=root, split='train', download=False, transform=train_transforms) 
        # train = datasets.ImageFolder(root = os.path.join(root, 'train'), 
        root_train = os.path.join(root, 'train') 
        train = ImageDataset(root = root_train, reader = '', class_map = '', load_bytes = False, transform = train_transforms) 
        # train = datasets.ImageNet(os.path.join(root, 'train'), download = False, transform=train_transforms) 
        root_val = os.path.join(root, 'val') 
        test = ImageDataset(root = root_val, reader = '', class_map = '', load_bytes = False, transform = test_transforms) 
        # test = datasets.ImageNet(os.path.join(root, 'val'), download=False, transform=test_transforms) 
        super(ImageNet2, self).__init__(train = train, test = test) 
