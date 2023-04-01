import torch
from typing import Tuple
import numpy as np
import h5py as h5
import cv2
import matplotlib.pyplot as plt

def get_od(H,W, K,c2w) -> Tuple[torch.tensor,torch.tensor]:
    """Get rays for each pixel.
    Args:
        H (int): image height
        W (int): image width
        K (torch.tensor): camera intrinsics
    Returns:
        rays_o (np.array): origin of rays, (H*W, 3)
        rays_d (np.array): direction of rays, (H*W, 3)
    """
    device="cuda" if torch.cuda.is_available() else "cpu"
    i, j = torch.meshgrid(torch.linspace(0,W-1,W), torch.linspace(0,H-1,H), indexing='xy')
    # Get direction of rays, going theough a pixel, first calcuate ICW->CCW
    # K=[[fx, 0, cx],
    #    [0, fy, cy], 
    #    [0, 0, 1]]
    i,j=i.to(device),j.to(device)
    i = ((i - K[0,2]) / K[0,0] ).reshape(-1)
    j = ((j - K[1,2]) / K[1,1] ).reshape(-1)
    # transform to camera coordinates: ICW->CCW (only direction matters)
    dirs = torch.stack((i, -j, -torch.ones_like(i)), axis=-1)
    # rays_d = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    print("Time taking step 0")
    rays_d=dirs@c2w[:3,:3]
    rays_o = (c2w[:3, 3:4].mT).expand((rays_d.shape[0]),-1)
    # transform to world coordinates
    # rays_d = torch.einsum('...ij,...j->...i', K, rays_d)
    # rays_o = torch.einsum('...ij,...j->...i', K, rays_o)
    return rays_o, rays_d/torch.norm(rays_d,dim=-1,keepdim=True)

def strat_sampler(
        tn:torch.tensor,
        tf:torch.tensor,
        num_samples:int,
)->torch.tensor:
    """Stratified sampling along a ray.
    Args:
        tn (torch.tensor): near depth
        tf (torch.tensor): far depth
        num_samples (int): number of samples
    Returns:
        t (torch.tensor): sampled points
    """
    device="cuda" if torch.cuda.is_available() else "cpu"

    t=torch.linspace(tn,tf,num_samples).to(device)
    t=t+(torch.rand_like(t)*(tf-tn)/num_samples).to(device)
    return t

def model(rays:torch.tensor)->torch.tensor:
    """NeRF model.
    Args:
        rays (torch.tensor): rays, (H*W, 3)
    Returns:
        model_out (torch.tensor): model output, (H*W, 4)
    """
    device="cuda" if torch.cuda.is_available() else "cpu"

    f = h5.File('datacube.hdf5', 'r')
    datacube = np.array(f['density'])
    datacube=torch.from_numpy(datacube).to(device)
    model_out=torch.rand((rays.shape[0],rays.shape[1],4)).to(device)
    print("MIN_MAXES:",rays.min(),rays.max())
    return model_out

def vol_render(
        # model_out:torch.Tensor,
        rays_d:torch.Tensor,
        rays_o:torch.Tensor,
        t:torch.Tensor=None,
        near:float=0.0,
        far:float=1.0,
        num_samples=100,
    )->Tuple[torch.Tensor,torch.Tensor]:
    device="cuda" if torch.cuda.is_available() else "cpu"

    if t is None:
        t=strat_sampler(near,far,num_samples)
        # t=t[None,:,None]
    # print("t_shape:",rays_o[:,None,:].shape)
    print("Time_ttaking1:")
    rays=rays_o[:,None,:]+rays_d[:,None,:]*t[None,:,None]
    model_out=model(rays)
    sigma=model_out[...,3]
    rgb=model_out[...,0:3]

    del_t=torch.zeros_like(t).to(device)

    del_t[...,1:]=t[...,1:]-t[...,:-1]
    del_t=del_t[None,:]
    print("Time_ttaking2:")

    print("rays_shape:",(del_t*sigma).shape)
    prod=sigma*del_t
    T=torch.exp(-torch.cumsum(prod,axis=-1))
    alpha=1-torch.exp(-prod)
    print("Shape:",T.shape,del_t.shape,alpha.shape,rgb.shape)
    C=torch.sum(T[:,:,None]*alpha[:,:,None]*rgb,dim=-2)
    print(C.shape)
    return C
model_out=torch.rand((480,620,4))


# print(datacube.shape)
device='cuda' if torch.cuda.is_available() else 'cpu'
print("deivce:",device)
H=480
W=620

K=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]])).to(device)
c2w=torch.randint(5,(4,4)).to(torch.float32).to(device)
rays_o, rays_d=get_od(H,W,K,c2w)
C=vol_render(rays_o,rays_d,num_samples=100)
C=C.reshape(H,W,3).cpu().numpy()
plt.imshow(C)

plt.show()