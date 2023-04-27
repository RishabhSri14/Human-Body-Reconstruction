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
import argparse
from dataset import NeRF_DATA

parser = argparse.ArgumentParser(description='Train Hashing')
parser.add_argument('--display',action='store_true',help='Display the output')
parser.add_argument('--compile',action='store_true',help='Use torch.compile(), might speed up')
parser.add_argument('--load',action='store_true',help='Continue from checkpoint')
parser.add_argument('--update_rate',type=int,default=15,help='Update rate for Occupancy grid')
parser.add_argument('--write',action='store_true',help='Write the output')
parser.add_argument('--num_epochs',type=int,default=1000,help='Number of epochs')
parser.add_argument('--num_batch',type=int,default=1000,help='Ray batch size')
# print(datacube.shape)
args=parser.parse_args()
device='cuda' if torch.cuda.is_available() else 'cpu'
print("deivce:",device)
K=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]])).to(device)
num_freq=3

num_epoch=args.num_epochs
# data = np.load('tiny_nerf_data.npz')
# images = (data['images'])
# poses = (data['poses'])
# focal = torch.from_numpy(data['focal'])
# train_imgs=torch.from_numpy(images[:-1,...])
# # test_imgs=torch.from_numpy(images[-1:-2:-1,...].copy())
# test_imgs=torch.from_numpy(images[101:102,...].copy())
# train_pose=torch.from_numpy(poses[:-1,...])
# # test_pose=torch.from_numpy(poses[-1:-2:-1,...].copy())
# test_pose=torch.from_numpy(poses[101:102,...].copy())
# # H,W,_=images.shape[1:]
#####################TEST#############################
# Ht,Wt,_=test_imgs.shape[1:]
# K_test=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]])).to(device)
# K_test[0,0]=focal
# K_test[1,1]=focal
# K_test[0,2]=Wt/2  
# K_test[1,2]=Ht/2
# c2w_test=test_pose.squeeze(0).to(device)
# rays_o, rays_d=get_od(Ht,Wt,K_test,c2w_test)
# test_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(rays_o.cpu(),rays_d.cpu()),batch_size=10000,shuffle=False,num_workers=4,pin_memory=True)

#################Train DataLoader#####################
pth_train='data/lego/transforms_train.json'
train_data=NeRF_DATA(json_path=pth_train)
train_loader_nerf=torch.utils.data.DataLoader(train_data,batch_size=2,shuffle=True,num_workers=8,pin_memory=True)
train_data.focal
K=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]]))

K[0,0]=train_data.focal
K[1,1]=train_data.focal
K[0,2]=train_data.W/2
K[1,2]=train_data.H/2
H,W=train_data.H,train_data.W

pth_test='data/lego/transforms_tmp.json'
test_data=NeRF_DATA(json_path=pth_test)
test_loader_nerf=torch.utils.data.DataLoader(test_data,batch_size=1,shuffle=False,num_workers=8,pin_memory=True)


# Ht,Wt,_=train_data[0][0].shape
# K_test=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]])).to(device)
# K_test[0,0]=train_data.focal
# K_test[1,1]=train_data.focal
# K_test[0,2]=train_data.W/2  
# K_test[1,2]=train_data.H/2
# c2w_test=train_data[50][1].unsqueeze(0).to(device)
# rays_o, rays_d=get_od(train_data.H,train_data.W,K_test,c2w_test)
# test_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(rays_o.unsqueeze(0).cpu(),rays_d),batch_size=1,shuffle=False,num_workers=4,pin_memory=True)

######################################################

L=16
F=2
max_bound,min_bound=find_bounding_box(train_loader_nerf,near=2.0,far=6.0,K=K)
print("BOUNDING BOX:",max_bound,min_bound)
mu=min_bound.to(device)

# sigma=((max_bound-min_bound)**2).sum().sqrt().to(device)
sigma=(torch.abs(max_bound-min_bound))
# encoder=HashEncoder(N_min=16,N_max=2**10,L=L,F=F,T=2**16,dim=3,mu=mu,sigma=sigma)
encoder=HashEncoder(N_min=16,N_max=2**14,L=L,F=F,T=2**20,dim=3,mu=mu,sigma=sigma)
dir_encoder=PositionalEncoder(d_model=3,num_freq=num_freq)
VolumeRenderer=Volume_Renderer(H=H,W=W,K=K,near=2.,far=6.,device=device,Pos_encode=encoder,Dir_encode=dir_encoder,max_dim=2**10,sigma_val=sigma,mu=mu)

nerf=torch.nn.DataParallel(MLP_3D(num_sig=2,num_col=2,L=L,F=F,d_view=3*num_freq*2))
# if args.load is True:
if args.load is True:
    nerf.load_state_dict(torch.load('Nerf_hash_good.pth'))
    encoder.load_state_dict(torch.load('encoder_hash_good.pth'))

nerf=nerf.to(device)
encoder=encoder.to(device)
dir_encoder=dir_encoder.to(device)
if args.compile is True:
    nerf=torch.compile(nerf,mode='reduce-overhead')
# encoder=torch.compile(encoder,mode='max-autotune')

optimizer_embed=torch.optim.SparseAdam(list(encoder.Embedding_list.parameters()),lr=0.01)
optimizer_MLP=torch.optim.AdamW(nerf.parameters(),lr=0.01)
criterion=torch.nn.MSELoss()

scheduler_embed = torch.optim.lr_scheduler.OneCycleLR(optimizer_embed, 
                    max_lr = 5e-2, # Upper learning rate boundaries in the cycle for each parameter group
                    steps_per_epoch = len(train_loader_nerf)*80, # The number of steps per epoch to train for.
                    epochs = num_epoch, # The number of epochs to train for.
                    anneal_strategy = 'cos') 
scheduler_MLP = torch.optim.lr_scheduler.OneCycleLR(optimizer_MLP, 
                       max_lr = 5e-1, # Upper learning rate boundaries in the cycle for each parameter group
                       steps_per_epoch = len(train_loader_nerf)*80, # The number of steps per epoch to train for.
                       epochs = num_epoch, # The number of epochs to train for.
                       anneal_strategy = 'cos') 
# scheduler_MLP= torch.optim.lr_scheduler.CyclicLR(optimizer_MLP, 
#                      base_lr = 0.01, # Initial learning rate which is the lower boundary in the cycle for each parameter group
#                      max_lr = 0.1, # Upper learning rate boundaries in the cycle for each parameter group
#                      step_size_up = 1000, # Number of training iterations in the increasing half of a cycle
#                      mode = "exp_range",cycle_momentum=False)

criterion=nn.MSELoss()
loss=0
i=0
display=args.display
write_img=args.write
nerf.train()
encoder.train()
n_imgs=3


# NOTE Mixed Precision Scaler Here
p=0
scaler=torch.cuda.amp.GradScaler()
for epoch in range(num_epoch):
    pbar= tqdm(enumerate(train_loader_nerf),desc=f"Train:{i}:{loss}",total=len(train_loader_nerf))
    for i,batch in pbar:
        t1=time.time()
        image,c2w,_=batch
        image=image
        c2w=c2w
        rays_o, rays_d=get_od(H,W,K,c2w)

        gts=image.permute(0,2,3,1)
        gts=gts.reshape(-1,3)
        # print(rays_o.shape,rays_d.shape)
        rays_o,rays_d=rays_o.reshape(-1,3),rays_d.reshape(-1,3)
        print("rays_o SHAPE!!!",rays_o.shape)
        train_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(rays_o,rays_d,gts),batch_size=16000,shuffle=True,num_workers=8,pin_memory=True)

        t1=time.time()-t1

        t21=time.time()
        p+=1
        if (p)%args.update_rate==0:
            VolumeRenderer.reset_mask=True
            update_mask=True
            print("Updating Mask")
            p=0
        else:
            update_mask=False
        for ray_o,ray_d,gt in tqdm(train_loader,leave=False):
            with torch.cuda.amp.autocast():
                ray_o=ray_o.to(device)
                ray_d=ray_d.to(device)
                gt=gt.to(device)
                # C=vol_render(nerf,ray_d,ray_o,near=2.,far=6.,num_samples=32,Pos_encode=encoder,Dir_encode=dir_encoder)
                C=VolumeRenderer.vol_render(nerf,ray_d,ray_o,num_samples=32,update_mask=update_mask)
                t11=time.time()
                loss=criterion(C,gt)#/len(train_loader)
                scaler.scale(loss).backward()
                t11=time.time()-t11
                scaler.step(optimizer_embed)
                scaler.step(optimizer_MLP)
                scheduler_embed.step()
                scheduler_MLP.step()
                optimizer_MLP.zero_grad(set_to_none=True)
                optimizer_embed.zero_grad(set_to_none=True)
                scaler.update()
            # print("time:",t11)
                # Color.append(C)
            # pred[prev_len:prev_len+C.shape[0]]=C
            # prev_len=C.shape[0]
        # loss.backward()
        t21=time.time()-t21
        t22=time.time()
        # optimizer_embed.step()
        # optimizer_MLP.step()
       
        # print("Time_2:",time.time()-t1)
        t22=time.time()-t22
        # if display==True :#and int(100*i/num_epoch)%1==0 and np.ceil(100*i/num_epoch)==np.floor(100*i/num_epoch):
        #     img=pred.reshape(H,W,3)
        #     img_np=img.detach().cpu().numpy()
        #     cv2.imshow("Output",img_np[...,::-1])
        #     key=cv2.waitKey(1)
        #     if key==ord('q'):
        #         exit(0)
        t3=time.time()
        if display==True:# and int(100*i/num_epoch)%1==0 and np.ceil(100*i/num_epoch)==np.floor(100*i/num_epoch):
            with torch.no_grad():
                # rays_o_batch=make_batch(rays_o,batch_size=100)
                # rays_d_batch=make_batch(rays_d,batch_size=100)
                pred=torch.zeros(H*W,3,device=device)
                # for ray_o,ray_d in zip(rays_o_batch,rays_d_batch):
                prev_len=0
                for ray_o,ray_d in test_loader:
                    ray_o=ray_o.to(device)
                    ray_d=ray_d.to(device)
                    # C=vol_render(nerf,ray_d,ray_o,near=2.,far=6.,num_samples=64,Pos_encode=encoder,Dir_encode=dir_encoder)
                    C=VolumeRenderer.vol_render(nerf,ray_d,ray_o,num_samples=32)
                    pred[prev_len:prev_len+C.shape[0]]=C
                    prev_len=C.shape[0]
                # pred=torch.cat(Color,dim=0)
                img=pred.reshape(H,W,3)
                img_np=img.detach().cpu().numpy()
                # img_out=(pred.reshape(H,W,3).detach().cpu().numpy())
                cv2.imshow("Output",img_np[...,::-1])
                key=cv2.waitKey(1)
                if key==ord('q'):
                    exit(0)
        prev_len=0
        if write_img==True and int(100*i/num_epoch)%1==0 and np.ceil(100*i/num_epoch)==np.floor(100*i/num_epoch):
            with torch.no_grad():
                pred=torch.zeros(H*W,3,device=device)
                print(len(test_loader_nerf))
                for batch in test_loader_nerf:
                    image,c2w,_=batch
                    c2w=c2w
                    # c2w=pose.to(device)
                    rays_o, rays_d=get_od(H,W,K,c2w)
                    rays_o,rays_d=rays_o.reshape(-1,3),rays_d.reshape(-1,3)
                    test_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(rays_o,rays_d),batch_size=32000,shuffle=False,num_workers=8,pin_memory=True)
                    for ray_o, ray_d in tqdm(test_loader): 
                        ray_o=ray_o.to(device)
                        ray_d=ray_d.to(device)

                        C=VolumeRenderer.vol_render(nerf,ray_d,ray_o,num_samples=32,update_mask=True)
                        pred[prev_len:prev_len+C.shape[0]]=C
                        prev_len+=C.shape[0]
                    # break
                img=pred.reshape(H,W,3)
                img_np=img.detach().cpu().numpy()
                cv2.imwrite(f'./results/hash_big{epoch}_{i}.png',((img_np[...,::-1]-img_np.min())/(img_np.max()-img_np.min())*255).astype(np.uint8))
                # plt.imsave(f'./results/{i}.png',img_out)
                torch.save(nerf.state_dict(),'Nerf_hash.pth')
                torch.save(encoder.state_dict(),'encoder_hash.pth')
        torch.cuda.empty_cache()


    t3=time.time()-t3
    # print("time_3",time.time()-t1)
    pbar.desc=f'Train:{i}:{loss}, Epoch:{epoch}'
