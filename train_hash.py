import torch
from typing import Tuple,Optional
import numpy as np
import h5py as h5
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from vol_renderer import *
from encoder import *
from hash_encoding import *
from test_hash import *
from tqdm import tqdm
import time
# print(datacube.shape)
device='cuda' if torch.cuda.is_available() else 'cpu'
print("deivce:",device)
K=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]])).to(device)
num_freq=3
L=16
F=2
dir_encoder=PositionalEncoder(d_model=3,num_freq=num_freq)

encoder=HashEncoder(N_min=16,N_max=2**14,L=L,F=F,T=2**20,dim=3)
nerf=torch.nn.DataParallel(MLP_3D(num_sig=1,num_col=2,L=L,F=F,d_view=3*num_freq*2))
nerf=nerf.to(device)
encoder=encoder.to(device)

num_epoch=1000
data = np.load('tiny_nerf_data.npz')
images = (data['images'])
poses = (data['poses'])
focal = torch.from_numpy(data['focal'])
train_imgs=torch.from_numpy(images[:-1,...])
# test_imgs=torch.from_numpy(images[-1:-2:-1,...].copy())
test_imgs=torch.from_numpy(images[101:102,...].copy())
train_pose=torch.from_numpy(poses[:-1,...])
# test_pose=torch.from_numpy(poses[-1:-2:-1,...].copy())
test_pose=torch.from_numpy(poses[101:102,...].copy())
# H,W,_=images.shape[1:]
#####################TEST#############################
Ht,Wt,_=test_imgs.shape[1:]
K_test=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]])).to(device)
K_test[0,0]=focal
K_test[1,1]=focal
K_test[0,2]=Wt/2  
K_test[1,2]=Ht/2
c2w_test=test_pose.squeeze(0).to(device)
rays_o, rays_d=get_od(Ht,Wt,K_test,c2w_test)
test_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(rays_o.cpu(),rays_d.cpu()),batch_size=10000,shuffle=False,num_workers=4,pin_memory=True)

######################################################


optimizer_embed=torch.optim.SparseAdam(list(encoder.Embedding_list.parameters()),lr=0.2)
optimizer_MLP=torch.optim.AdamW(nerf.parameters(),lr=0.2)
criterion=torch.nn.MSELoss()

scheduler_embed = torch.optim.lr_scheduler.OneCycleLR(optimizer_embed, 
                    max_lr = 2e-1, # Upper learning rate boundaries in the cycle for each parameter group
                    steps_per_epoch = 1000, # The number of steps per epoch to train for.
                    epochs = num_epoch, # The number of epochs to train for.
                    anneal_strategy = 'cos') 
scheduler_MLP = torch.optim.lr_scheduler.OneCycleLR(optimizer_MLP, 
                       max_lr = 2e-1, # Upper learning rate boundaries in the cycle for each parameter group
                       steps_per_epoch = 1000, # The number of steps per epoch to train for.
                       epochs = num_epoch, # The number of epochs to train for.
                       anneal_strategy = 'cos') 

criterion=nn.MSELoss()
loss=0
i=0
display=True
nerf.train()
encoder.train()
n_imgs=3
pbar= tqdm(range(num_epoch),desc=f"Train:{i}:{loss}")
for i in pbar:
    optimizer_MLP.zero_grad(set_to_none=True)
    optimizer_embed.zero_grad(set_to_none=True)
    # rays_o=torch.tensor([],device=device)
    # rays_d=torch.tensor([],device=device)
    # gt=torch.tensor([],device=device)
    # for i in range(n_imgs):
    #     img_idxs=torch.randint(0,train_imgs.shape[0],(1,))
    #     image=train_imgs[img_idxs]
    #     pose=train_pose[img_idxs].squeeze(0)
    #     H,W,_=image.shape[1:]
    #     K=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]])).to(device)
    #     K[0,0]=focal
    #     K[1,1]=focal
    #     K[0,2]=W/2
    #     K[1,2]=H/2
    #     gt_t=image.reshape(-1,3).to(device)
    #     gt=torch.cat((gt,gt_t),dim=0)
    #     c2w=pose.to(device)
    #     rays_ot, rays_dt=get_od(H,W,K,c2w)
    #     rays_o=torch.cat((rays_o,rays_ot),dim=0)
    #     rays_d=torch.cat((rays_d,rays_dt),dim=0)
    t1=time.time()
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
    train_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(rays_o.cpu(),rays_d.cpu()),batch_size=10000,shuffle=False,num_workers=4,pin_memory=True)
    gt=image.reshape(-1,3).to(device)
    # rays_d=torch.dataloader(torch.data.TensorDataset(rays_d),batch_size=512,shuffle=True,num_workers=4,pin_memory=True)
    # rays_o_batch=make_batch(rays_o,batch_size=256)
    # rays_d_batch=make_batch(rays_d,batch_size=256)
    t1=time.time()-t1
    Color=[]
    # for ray_o,ray_d in zip(rays_o_batch,rays_d_batch):
    t2=time.time()
    for ray_o,ray_d in train_loader:
        ray_o=ray_o.to(device)
        ray_d=ray_d.to(device)
        C=vol_render(nerf,ray_d,ray_o,near=2.,far=6.,num_samples=64,Pos_encode=encoder,Dir_encode=dir_encoder)
        Color.append(C)
    pred=torch.cat(Color,dim=0)
    # gt=torch.from_numpy(image).to(device).reshape(-1,3)
    #print("max",gt.max())
    loss=criterion(pred,gt)
    loss.backward()
    optimizer_embed.step()
    optimizer_MLP.step()
    scheduler_embed.step()
    scheduler_MLP.step()
    # print("Time_2:",time.time()-t1)
    t2=time.time()-t2
    # if display==True :#and int(100*i/num_epoch)%1==0 and np.ceil(100*i/num_epoch)==np.floor(100*i/num_epoch):
    #     img=pred.reshape(H,W,3)
    #     img_np=img.detach().cpu().numpy()
    #     cv2.imshow("Output",img_np[...,::-1])
    #     key=cv2.waitKey(1)
    #     if key==ord('q'):
    #         exit(0)
    t3=time.time()
    if display==True: #and int(100*i/num_epoch)%1==0 and np.ceil(100*i/num_epoch)==np.floor(100*i/num_epoch):
        with torch.no_grad():
            # rays_o_batch=make_batch(rays_o,batch_size=100)
            # rays_d_batch=make_batch(rays_d,batch_size=100)
            Color=[]
            # for ray_o,ray_d in zip(rays_o_batch,rays_d_batch):
            for ray_o,ray_d in test_loader:
                ray_o=ray_o.to(device)
                ray_d=ray_d.to(device)
                C=vol_render(nerf,ray_d,ray_o,near=2.,far=6.,num_samples=128,Pos_encode=encoder,Dir_encode=dir_encoder)
                Color.append(C)
            pred=torch.cat(Color,dim=0)
            img=pred.reshape(H,W,3)
            img_np=img.detach().cpu().numpy()
            # img_out=(pred.reshape(H,W,3).detach().cpu().numpy())
            cv2.imshow("Output",img_np[...,::-1])
            key=cv2.waitKey(1)
            if key==ord('q'):
                exit(0)
            #cv2.imwrite(f'./results/hash{i}.png',((img_np[...,::-1]-img_np.min())/(img_np.max()-img_np.min())*255).astype(np.uint8))
            # plt.imsave(f'./results/{i}.png',img_out)
            # torch.save(nerf.state_dict(),'Nerf.pth')
    t3=time.time()-t3
    # print("time_3",time.time()-t1)
    pbar.desc=f'Train:{i}:{loss}, Time:{t1}, Time_2:{t2}, Time_3:{t3}'
