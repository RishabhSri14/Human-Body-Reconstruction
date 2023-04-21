from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
import random


def collate_function(batch):
    images = torch.stack([x[0] for x in batch])
    transform_mat = torch.stack([x[1] for x in batch])
    rotation = torch.tensor([x[2] for x in batch])

    # (batch_size, channels, h, w),  (batch_size,4,4), batch_size
    return (images, transform_mat, rotation)


def make_data_loader(dataset,
                     batch_size,
                     num_workers,
                     sampler=None):

    return DataLoader(dataset,
                      batch_size=batch_size,
                      collate_fn=lambda x: collate_function(x),
                      num_workers=num_workers,
                      persistent_workers=True,
                      sampler=sampler)

