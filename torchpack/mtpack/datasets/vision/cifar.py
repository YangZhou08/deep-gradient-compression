import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset

from .utils import split_train_val_indices
from ..dataset import Dataset

__all__ = ['CIFAR']


class CIFAR(Dataset):
    def __init__(self, root, num_classes, image_size, val_ratio=None, extra_train_transforms=None):
        if num_classes == 10:
            dataset = datasets.CIFAR10
        elif num_classes == 100:
            dataset = datasets.CIFAR100
        else:
            raise NotImplementedError('only support CIFAR10/100 for now')

        train_transforms_pre = [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        train_transforms_post = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
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
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
        ]
        test_transforms = transforms.Compose(test_transforms)

        train = dataset(root=root, train=True, download=True, transform=train_transforms)
        test = dataset(root=root, train=False, download=True, transform=test_transforms)

        if val_ratio is None:
            super().__init__(train=train, test=test)
        else:
            train_indices, val_indices = split_train_val_indices(
                targets=train.targets, val_ratio=val_ratio, num_classes=num_classes
            )
            train = Subset(train, indices=train_indices)
            val = Subset(dataset(root=root, train=True, download=True, transform=test_transforms), indices=val_indices)
            super().__init__(train=train, val=val, test=test)
