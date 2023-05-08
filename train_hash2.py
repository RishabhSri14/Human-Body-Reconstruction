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
from dataset_new import NeRF_DATA_NEW
from helper import *
from tmp_encoder import *
import os
parser = argparse.ArgumentParser(description='Train Hashing')
parser.add_argument('--display',action='store_true',help='Display the output')
parser.add_argument('--compile',action='store_true',help='Use torch.compile(), might speed up')
parser.add_argument('--load',action='store_true',help='Continue from checkpoint')
parser.add_argument('--update_rate',type=int,default=15,help='Update rate for Occupancy grid')
parser.add_argument('--write',action='store_true',help='Write the output')
parser.add_argument('--num_epochs',type=int,default=1000,help='Number of epochs')
parser.add_argument('--num_batch',type=int,default=16000,help='Ray batch size')
parser.add_argument('--num_imgs',type=int,default=2,help='Number of imgs in a batch')
parser.add_argument('--num_samples',type=int,default=64,help='Number of samples along ray')
parser.add_argument('--near',type=float,default=2.0,help='Near point')
parser.add_argument('--far',type=float,default=6.0,help='Far point')
parser.add_argument('--plot_grads',action='store_true',help='Plot the gradients after each iteration')
parser.add_argument('--use_sdf',action='store_true',help='Use sdf formulation while training')
parser.add_argument('--hierarchical',action='store_true',help='Use hierarchical sampling')
parser.add_argument('--max_res',type=float,default=2048,help='Max resolution of the grid')
parser.add_argument('--hash_size',type=float,default=16,help='Log Size of the hash table')
parser.add_argument("--model_name",type=str,default='default',help='Name of saved model')
parser.add_argument("--data_path",type=str,default=None,help='Path to data')
parser.add_argument("--ckpt_name",type=str,default='N_2048_T_16',help='Name of checkpoint')

# print(datacube.shape)
args=parser.parse_args()
device='cuda' if torch.cuda.is_available() else 'cpu'
print("deivce:",device)
K=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]])).to(device)
num_freq=4

num_epoch=args.num_epochs
#################Train DataLoader#####################
if args.data_path is None:
    pth_train_dir='data/lego/'
    pth_test_dir='data/lego/'
else:
    pth_train_dir=args.data_path
    pth_test_dir=args.data_path
pth_train=os.path.join(pth_train_dir,'transforms_train.json')
pth_test=os.path.join(pth_test_dir,"transforms_tmp.json")
if args.data_path is None:
    train_data=NeRF_DATA(json_path=pth_train)
    test_data=NeRF_DATA(json_path=pth_test)
else:
    train_data=NeRF_DATA_NEW(json_path=pth_train)
    test_data=NeRF_DATA_NEW(json_path=pth_test)

train_loader_bounds=torch.utils.data.DataLoader(train_data,batch_size=2,shuffle=True)
train_loader_nerf=torch.utils.data.DataLoader(train_data,batch_size=50,shuffle=True)
K=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]]))

K[0,0]=train_data.focal1
K[1,1]=train_data.focal2
K[0,2]=train_data.cx
K[1,2]=train_data.cy
H,W=int(train_data.H),int(train_data.W)
rays_o_list=[]
rays_d_list=[]
dir_norm_list=[]
gt_list=[]
for batch in tqdm(train_loader_nerf):
    t1=time.time()
    image,c2w,_=batch
    rays_o,rays_d,dir_norms=get_od(H,W,K,c2w)
    gts=image.permute(0,2,3,1)
    gts=gts.reshape(-1,3)
    # print(rays_o.shape,rays_d.shape)
    rays_o_lcl,rays_d_lcl=rays_o.reshape(-1,3),rays_d.reshape(-1,3)
    dir_norms=dir_norms.reshape(-1,1)
    rays_o_list.append(rays_o_lcl)
    rays_d_list.append(rays_d_lcl)
    dir_norm_list.append(dir_norms)
    gt_list.append(gts)
rays_o=torch.cat(rays_o_list,dim=0).to('cpu')
rays_d=torch.cat(rays_d_list,dim=0).to('cpu')
dir_norms =torch.cat(dir_norm_list,dim=0).to('cpu')
gts=torch.cat(gt_list,dim=0).to('cpu')
print("GROUND_TRUTH MIN_MAX",gts.max(),gts.min())
train_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(rays_o,rays_d,dir_norms,gts),batch_size=args.num_batch,shuffle=True,num_workers=8,pin_memory=True)
# for batch in tqdm(train_loader):
#     pass
print("SHAPES:",rays_o.shape,rays_d.shape,dir_norms.shape)


test_loader_nerf=torch.utils.data.DataLoader(test_data,batch_size=1,shuffle=False,num_workers=0,pin_memory=False)

######################################################

L=16
F=2
N_max=args.max_res
T=int(2**args.hash_size)
print("T:",T)
# max_bound,min_bound=find_bounding_box2(train_loader,near=torch.as_tensor(2.0),far=torch.as_tensor(6.0),K=K)
near=torch.tensor(args.near)
far=torch.tensor(args.far)
max_bound,min_bound=find_bounding_box(train_loader_bounds,near=near,far=far,K=K)
np.save('bounds_model.npy',torch.stack([min_bound,max_bound]).numpy())
print("BOUNDING BOX:",max_bound,min_bound)
mu=min_bound.to(device)

sigma=((max_bound-min_bound)**2).sum().sqrt().to(device)
encoder=HashEncoder(N_min=16,N_max=N_max,L=L,F=F,T=T,dim=3,mu=mu,sigma=sigma)
dir_encoder=PositionalEncoder(d_model=3,num_freq=num_freq)
var_model=None
if args.use_sdf is True:
    var_model=VarModel()
    var_model=var_model.to(device)
VolumeRenderer=Volume_Renderer(H=H,W=W,K=K,near=near,far=far,device=device,Pos_encode=encoder,Dir_encode=dir_encoder,max_dim=2**10,sigma_val=sigma,mu=mu,use_sdf=args.use_sdf,var_model=var_model)
nerf=torch.nn.DataParallel(MLP_3D(num_sig=2,num_col=2,L=L,F=F,d_view=3*num_freq*2,max_bound=max_bound,min_bound=min_bound))
# if args.load is True:
if args.load is True:
    nerf_ckpt=args.ckpt_name+'_Nerf_hash.pth'
    encoder_ckpt=args.ckpt_name+'_encoder_hash.pth'
    nerf.load_state_dict(torch.load(nerf_ckpt))
    encoder.load_state_dict(torch.load(encoder_ckpt))

nerf=nerf.to(device)
encoder=encoder.to(device)
dir_encoder=dir_encoder.to(device)
if args.compile is True:
    nerf=torch.compile(nerf,mode='reduce-overhead')

optimizer_embed=torch.optim.Adam(list(encoder.Embedding_list.parameters()),lr=0.05)
optimizer_MLP=torch.optim.AdamW(nerf.parameters(),lr=0.005)
criterion=torch.nn.HuberLoss(reduction='mean')

# scheduler_embed = torch.optim.lr_scheduler.OneCycleLR(optimizer_embed, 
#                     max_lr = 1e-2, # Upper learning rate boundaries in the cycle for each parameter group
#                     steps_per_epoch = len(train_loader), # The number of steps per epoch to train for.
#                     epochs = num_epoch, # The number of epochs to train for.
#                     anneal_strategy = 'cos') 
# scheduler_MLP = torch.optim.lr_scheduler.OneCycleLR(optimizer_MLP, 
#                        max_lr = 1e-2, # Upper learning rate boundaries in the cycle for each parameter group
#                        steps_per_epoch = len(train_loader), # The number of steps per epoch to train for.
#                        epochs = num_epoch, # The number of epochs to train for.
#                        anneal_strategy = 'cos') 

scheduler_embed = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_embed,
                              T_max = num_epoch*len(train_loader), # Maximum number of iterations.
                             eta_min = 1e-4)

scheduler_MLP = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_MLP,
                                T_max = num_epoch*len(train_loader), # Maximum number of iterations.
                                eta_min = 1e-4)

if args.use_sdf is True:
    optimizer_var=torch.optim.AdamW(var_model.parameters(),lr=0.01)
    scheduler_var = torch.optim.lr_scheduler.OneCycleLR(optimizer_var, 
                        max_lr = 5e-2, # Upper learning rate boundaries in the cycle for each parameter group
                        steps_per_epoch = len(train_loader), # The number of steps per epoch to train for.
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

num_samples=args.num_samples
# NOTE Mixed Precision Scaler Here
p=0
iters=len(train_loader)//100 # 100 is the number of images written
count_iter=0
print("ITERS PER IMAGE:",iters)
scaler=torch.cuda.amp.GradScaler()
for epoch in range(num_epoch):
    pbar= tqdm(enumerate(train_loader),desc=f"Train:{i}:{loss}",total=len(train_loader))
    print("DATASET_LENGTH:",len(train_loader))
    for i,batch in pbar:

        t1=time.time()-t1

        t21=time.time()
        p+=1
        if (p)%args.update_rate==0:
            VolumeRenderer.reset_mask=False
            update_mask=False
            # VolumeRenderer.reset_mask=True
            # update_mask=True
            print("Updating Mask")
            p=0
        else:
            update_mask=False
        ray_o,ray_d,dir_norm,gt=batch
        ray_o=ray_o.to(device)
        ray_d=ray_d.to(device)
        dir_norm=dir_norm.to(device)
        gt=gt.to(device)
        print("SHAPES:",ray_o.shape,ray_d.shape,gt.shape,dir_norm.shape)
            # C=vol_render(nerf,ray_d,ray_o,near=2.,far=6.,num_samples=32,Pos_encode=encoder,Dir_encode=dir_encoder)
        with torch.cuda.amp.autocast():
            t11=time.time()
            Cr,Cf,norm=VolumeRenderer.vol_render(nerf,ray_d,ray_o,num_samples=num_samples,update_mask=update_mask,dir_norm=dir_norm,hierarchical=args.hierarchical)
            loss=criterion(Cr,gt)+criterion(Cf,gt)#/len(train_loader)
            print("MAXPRED:",Cr[torch.argmax(Cr.mean())],gt[torch.argmax(Cr.mean())])
            if args.use_sdf is True:
                loss+=0.1*eikonal_loss(norm)
            t11=time.time()-t11
        scaler.scale(loss).backward()
        scaler.step(optimizer_embed)
        scaler.step(optimizer_MLP)
        if args.plot_grads is True:
            plot_grad_flow(encoder.Embedding_list.named_parameters())
        scheduler_embed.step()
        scheduler_MLP.step()
        optimizer_MLP.zero_grad(set_to_none=True)
        optimizer_embed.zero_grad(set_to_none=True)
        if args.use_sdf is True:
            scaler.step(optimizer_var)
            optimizer_var.zero_grad(set_to_none=True)
            scheduler_var.step()
        scaler.update()
        t21=time.time()-t21
        t22=time.time()

        # print("Time_2:",time.time()-t1)
        t22=time.time()-t22

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
                    C=VolumeRenderer.vol_render(nerf,ray_d,ray_o,num_samples=64,dir_norm=dir_norm)
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
        # if write_img==True and int(100*i/len(train_loader))%1==0 and np.ceil(100*i/len(train_loader))==np.floor(100*i/len(train_loader)):
        if write_img==True and count_iter==iters:
            count_iter=0
            with torch.no_grad():
                pred=torch.zeros(H*W,3,device=device)
                print("!!!!WRITING!!!!")
                print(len(test_loader_nerf))
                for batch in test_loader_nerf:
                    image,c2w,_=batch
                    # c2w=c2w[0:1,...]
                    # image=image[0:1,...]
                    rays_o, rays_d,dir_norms=get_od(H,W,K,c2w)
                    rays_o,rays_d=rays_o.reshape(-1,3),rays_d.reshape(-1,3)
                    dir_norms=dir_norms.reshape(-1,1)
                    test_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(rays_o,rays_d,dir_norms),batch_size=16000,shuffle=False,num_workers=8,pin_memory=True)
                    for ray_o, ray_d,dir_norm in tqdm(test_loader): 
                        ray_o=ray_o.to(device)
                        ray_d=ray_d.to(device)
                        dir_norm=dir_norm.to(device)
                        print(ray_o.shape,ray_d.shape,dir_norm.shape)
                        _,C,_=VolumeRenderer.vol_render(nerf,ray_d,ray_o,num_samples=256,update_mask=True,dir_norm=dir_norm,hierarchical=args.hierarchical)
                        pred[prev_len:prev_len+C.shape[0]]=C
                        prev_len+=C.shape[0]
                        print("C.shape!!",C.shape[0])
                    # break
                img=pred.reshape(H,W,3)
                img_np=img.detach().cpu().numpy()
                cv2.imwrite(f'./results/hash_big_diff{epoch}_{i}.png',((img_np[...,::-1]-img_np.min())/(img_np.max()-img_np.min())*255).astype(np.uint8))
                # plt.imsave(f'./results/{i}.png',img_out)
                torch.save(nerf.state_dict(),f'{args.model_name}_Nerf_hash.pth')
                torch.save(encoder.state_dict(),f'{args.model_name}_encoder_hash.pth')
        torch.cuda.empty_cache()
        count_iter+=1

    t3=time.time()-t3
    # print("time_3",time.time()-t1)
    pbar.desc=f'Train:{i}:{loss}, Epoch:{epoch}'
