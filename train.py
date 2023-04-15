import torch
from typing import Tuple,Optional
import numpy as np
import h5py as h5
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from vol_renderer import *
from encoder import *
from tqdm import tqdm

# print(datacube.shape)
device='cuda' if torch.cuda.is_available() else 'cpu'
print("deivce:",device)
K=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]])).to(device)
num_freq=10
nerf=NeRF(d_input=3*num_freq*2,d_viewdirs=3*num_freq*2)
pos_enc=PositionalEncoder(d_model=3,num_freq=num_freq)
dir_enc=PositionalEncoder(d_model=3,num_freq=num_freq)
nerf=nerf.to(device)

num_iters=1000
data = np.load('tiny_nerf_data.npz')
images = (data['images'])
# print(images.shape)
poses = (data['poses'])
focal = torch.from_numpy(data['focal'])
train_imgs=torch.from_numpy(images[:-1,...])
# test_imgs=torch.from_numpy(images[-1:-2:-1,...].copy())
test_imgs=torch.from_numpy(images[101:102,...].copy())
train_pose=torch.from_numpy(poses[:-1,...])
# test_pose=torch.from_numpy(poses[-1:-2:-1,...].copy())
test_pose=torch.from_numpy(poses[101:102,...].copy())
# H,W,_=images.shape[1:]

optimizer=torch.optim.Adam(nerf.parameters(),lr=1e-2)
# scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=int(num_iters*0.2),gamma=0.5)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=num_iters,eta_min=5e-4)
Loss=nn.MSELoss()
loss=0
i=0
for i in tqdm(range(num_iters),desc=f"Train:{i}:{loss}"):
    optimizer.zero_grad()
    
    img_idxs=torch.randint(0,train_imgs.shape[0],(1,))
    image=train_imgs[img_idxs]
    pose=train_pose[img_idxs].squeeze(0)
    H,W,_=image.shape[1:]
    K=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]])).to(device)
    K[0,0]=focal
    K[1,1]=focal
    K[0,2]=W/2
    K[1,2]=H/2

    c2w=pose.to(device)
    rays_o, rays_d=get_od(H,W,K,c2w)

    rays_o_batch=make_batch(rays_o,batch_size=256)
    rays_d_batch=make_batch(rays_d,batch_size=256)
    Color=[]
    for ray_o,ray_d in zip(rays_o_batch,rays_d_batch):
        ray_o=ray_o.to(device)
        ray_d=ray_d.to(device)
        C=vol_render(nerf,ray_d,ray_o,near=2.,far=6.,num_samples=64,Pos_encode=pos_enc,Dir_encode=dir_enc)
        Color.append(C)
    pred=torch.cat(Color,dim=0)
    # gt=torch.from_numpy(image).to(device).reshape(-1,3)
    gt=image.reshape(-1,3).to(device)
#    print("max",gt.max())
    loss=Loss(pred,gt)
    loss.backward()
    optimizer.step()
    scheduler.step()
    if int(100*i/num_iters)%5==0 and np.ceil(100*i/num_iters)==np.floor(100*i/num_iters):
        with torch.no_grad():
            H,W,_=test_imgs.shape[1:]
            K=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]])).to(device)
            K[0,0]=focal
            K[1,1]=focal
            K[0,2]=W/2  
            K[1,2]=H/2
            c2w=test_pose.squeeze(0).to(device)
            rays_o, rays_d=get_od(H,W,K,c2w)
            rays_o_batch=make_batch(rays_o,batch_size=100)
            rays_d_batch=make_batch(rays_d,batch_size=100)
            Color=[]
            for ray_o,ray_d in zip(rays_o_batch,rays_d_batch):
                ray_o=ray_o.to(device)
                ray_d=ray_d.to(device)
                C=vol_render(nerf,ray_d,ray_o,near=2.,far=6.,num_samples=64,Pos_encode=pos_enc,Dir_encode=dir_enc)
                Color.append(C)
            pred=torch.cat(Color,dim=0)
            img_out=(pred.reshape(H,W,3).cpu().numpy()*255).astype(np.uint8)
            plt.imsave(f'./results/{i}.png',img_out)
            torch.save(nerf.state_dict(),'Nerf.pth')
