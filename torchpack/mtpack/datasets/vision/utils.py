import torch

__all__ = ['split_train_val_indices']


def split_train_val_indices(targets, val_ratio, num_classes):
    assert 0 < val_ratio < 1
    train_size = len(targets)
    val_size = int(train_size * val_ratio)

    g = torch.Generator()
    g.manual_seed(180452398)
    indices = torch.randperm(train_size, generator=g).tolist()
    val_size_per_class = ([val_size // num_classes + 1] * (val_size % num_classes)
                          + [val_size // num_classes] * (num_classes - val_size % num_classes))

    train_indices, val_indices = [], []
    for idx in indices:
        if val_size_per_class[targets[idx]] > 0:
            val_indices.append(idx)
            val_size_per_class[targets[idx]] -= 1
        else:
            train_indices.append(idx)
    return train_indices, val_indices
