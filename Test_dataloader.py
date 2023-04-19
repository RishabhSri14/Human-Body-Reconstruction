from dataset import *
from dataloader import *

# Sample data=> (800,800,3), (4,4), 1
train_dataset = NeRF(json_path='./data/nerf_synthetic/nerf_synthetic/lego/transforms_train.json')
train_dataset = NeRF(json_path='./data/nerf_synthetic/nerf_synthetic/lego/transforms_val.json')
train_dataset = NeRF(json_path='./data/nerf_synthetic/nerf_synthetic/lego/transforms_test.json')

train_dataloader = make_data_loader(train_dataset, batch_size=8, num_workers=1)
batch=next(iter(train_dataloader))
img, transform_mat, rotation=batch
print(img.shape, transform_mat.shape, rotation.shape)
