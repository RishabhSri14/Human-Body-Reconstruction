import torch
from typing import Tuple
import time
from tqdm import tqdm
from typing import Tuple,Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import  Line2D
from torch import nn
from test_hash import MLP_3D
# from hash_encoding import HashEncoder

class VarModel(nn.Module):
  def __init__(self):
    super(VarModel, self).__init__()

    self.b = nn.Parameter(torch.tensor(0.5))

  def forward(self, x):
    phi = 1/(1+torch.exp(-x*self.b)) 
    return phi

def hierarchical_sampling(
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        z_vals: torch.Tensor,
        weights: torch.Tensor,
        n_samples: int,
        tn: float,
        tf: float,
        perturb: bool = False,
        device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t1=time.time()
    # weights=weights.clone()
    weights[weights<0]=0
    weights=weights.squeeze(-1)
    pdf=(weights+1e-5)/torch.sum(weights+1e-5,dim=-1,keepdims=True)
    cdf=torch.cumsum(pdf,dim=-1)
    u=torch.rand(cdf.shape,device=device)
    inds=torch.searchsorted(cdf,u,right=True)
    print("WTS_SHAPE:",weights.shape,cdf.shape,u.shape,inds.shape)
    samples=torch.rand(n_samples,device=device)*(tf-tn)+tn
    inds=torch.clamp(inds,min=0,max=samples.shape[-1]-1)
    samples=samples[inds]
    z_vals=z_vals.expand(list(inds.shape[:-1])+[z_vals.shape[-1]])
    combined_samples,_=torch.sort(torch.cat([z_vals,samples],dim=-1),dim=-1)
    rays=rays_o[...,None,:]+rays_d[...,None,:]*combined_samples[...,:,None]    
    t1=time.time()-t1
    print("Extra_time:",t1)
    return rays,combined_samples

def calc_color(
    t: torch.Tensor,
    rgb: torch.Tensor,
    sigma: torch.Tensor,
    dir_norm: torch.Tensor,
    use_sdf: bool = False,
    var_model: Optional[nn.Module] = None,
    rays: torch.Tensor = None,
    model:Optional[MLP_3D]=None,
    encoder:Optional[nn.Module]=None,
    device: str = 'cuda'
    ):
    del_t=torch.zeros_like(t,device=device)

    del_t[...,:-1]=t[...,1:]-t[...,:-1]
    # del_t[...,-1]=1e10
    if del_t.ndim==1:
        del_t=del_t[None,:]
    del_t=del_t*dir_norm

    # prod=sigma*del_t
    # prod=torch.nn.functional.relu(sigma-torch.randn_like(sigma)*1e-5)*del_t
    # prod=torch.nn.functional.relu(sigma)*del_t
    sigma[sigma<-10]=-10
    prod=sigma*del_t
    # alpha=1-torch.exp(-torch.nn.functional.relu(prod+torch.randn_like(prod)*0.001))
    norm=None
    if use_sdf:
        phi=var_model(sigma)
        alpha=torch.zeros_like(sigma,device=device)
        alpha[...,:-1]=1-phi[...,1:]/phi[...,:-1]
        alpha=torch.nn.functional.relu(alpha)
        T=cumprod_exclusive(1-alpha)
        
        grads=model.module.finite_difference_normals_approximator(rays,encoder=encoder)
        print("NORMAL_SHAPE:",grads.shape)
        norm=eikonal_value(grads)
    else:
        alpha=1-torch.exp(-prod)
        # alpha=1-torch.exp(-torch.nn.functional.relu(prod))
        T=torch.exp(-torch.cumsum(prod,axis=-1))
        T=torch.roll(T,1,dims=-1)
        T[...,0]=1
        # tmp=T[...,0]
        # T[...,:-1]=T[...,1:]
        # T[...,-1]=tmp
        # alpha=1-torch.exp(-torch.nn.functional.relu(prod))
        # print("BEFORE_SHAPES:",T.shape,alpha.shape)
        # print("ALL SHAPES:",T[:,:,None].shape,alpha[:,:,None].shape,rgb.shape,sigma.shape,del_t.shape)
    wts=T[:,:,None]*alpha[:,:,None]
    # print("REQUIRED_SHAPES:",T[:,:,None].shape,alpha[:,:,None].shape,rgb.shape)
    # print("MIN_MAXES INSIDE:",T.max(),T.min(),alpha.max(),alpha.min(),rgb.max(),rgb.min())
    Cr=torch.sum(T[:,:,None]*alpha[:,:,None]*rgb,dim=-2)

    return Cr,wts,norm

def find_bounding_box(data_loader,near,far,K,num_samples=64,exp=False,device=None):
    if device is None:
        device=K.device
    # K[0,0]=focal
    # K[1,1]=focal
    W=2*K[0,2]
    H=2*K[1,2]
    # K=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]])).to(device)
    if exp:
        t=torch.from_numpy(np.asarray([near,far*torch.exp(torch.as_tensor(torch.log(far)-torch.log(near))/num_samples)])).to(device)
    else:
        t=torch.from_numpy(np.asarray([near,far+1.5])).to(device)
    min_bound=torch.ones(3,device=device)*(1e7)
    max_bound=torch.ones(3,device=device)*(-1e7)
    with torch.no_grad():
        for batch in tqdm(data_loader,total=len(data_loader),desc="Bounding Box Calculation..."):
            # H,W=image.shape[:2]
            _,c2w,_=batch
            c2w=c2w.to(device)
            rays_o, rays_d,_ = get_od(H, W, K, c2w)
            # t=strat_sampler(near,far,2,device=device)
            # print("T_shape:",t.shape)
            rays=rays_o[...,None,:]+rays_d[...,None,:]*t[None,:,None]
            rays=rays.reshape(-1,3)

            for i in range(3):
                min_elem=torch.min(rays[:,i])
                max_elem=torch.max(rays[:,i])
                if min_bound[i]>min_elem:
                    min_bound[i]=min_elem
                if max_bound[i]<max_elem:
                    max_bound[i]=max_elem
    return max_bound,min_bound

def find_bounding_box2(data_loader,near,far,K,num_samples=64,exp=False,device=None):
    if device is None:
        device=K.device
    # K[0,0]=focal
    # K[1,1]=focal
    W=2*K[0,2]
    H=2*K[1,2]
    # K=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]])).to(device)
    if exp:
        t=torch.from_numpy(np.asarray([near,far*torch.exp(torch.as_tensor(torch.log(far)-torch.log(near))/num_samples)])).to(device)
    else:
        t=torch.from_numpy(np.asarray([near,far+1.5])).to(device)
    min_bound=torch.ones(3,device=device)*(1e7)
    max_bound=torch.ones(3,device=device)*(-1e7)
    with torch.no_grad():
        for batch in tqdm(data_loader,total=len(data_loader),desc="Bounding Box Calculation..."):
            # H,W=image.shape[:2]
            print("REACHED_HERE")
            rays_o, rays_d,_ ,_= batch
            # t=strat_sampler(near,far,2,device=device)
            # print("T_shape:",t.shape)
            rays=rays_o[...,None,:]+rays_d[...,None,:]*t[None,:,None]
            rays=rays.reshape(-1,3)

            for i in range(3):
                min_elem=torch.min(rays[:,i])
                max_elem=torch.max(rays[:,i])
                if min_bound[i]>min_elem:
                    min_bound[i]=min_elem
                if max_bound[i]<max_elem:
                    max_bound[i]=max_elem
    return max_bound,min_bound

def get_od(
        H,
        W,
        K,
        c2w:torch.tensor,
        find_inv: Optional [bool] = False,
        ) -> Tuple[torch.tensor,torch.tensor]:
    """Get rays for each pixel.
    Args:
        H (int): image height
        W (int): image width
        K (torch.tensor): camera intrinsics
    Returns:
        rays_o (np.array): origin of rays, (H*W, 3)
        rays_d (np.array): direction of rays, (H*W, 3)
    """
    device=c2w.device
    i,j=torch.meshgrid(torch.arange(W,device=device), torch.arange(H,device=device), indexing='xy')
    # Get direction of rays, going theough a pixel, first calcuate ICW->CCW
    # K=[[fx, 0, cx],
    #    [0, fy, cy], 
    #    [0, 0, 1]]
    i = ((i - K[0,2]) / K[0,0] ).reshape(-1)
    j = ((j - K[1,2]) / K[1,1] ).reshape(-1)
    # transform to camera coordinates: ICW->CCW (only direction matters)
    dirs = torch.stack((i, -j, -torch.ones_like(i)), axis=-1)
    if find_inv:
        mat=c2w[:3,:3]
        rays_d=(torch.linalg.inv(c2w[...,:3,:3])@dirs.mT).mT
    else:
        rays_d=((c2w[...,:3,:3])@(dirs.mT)).mT
    rays_o = (c2w[...,:3, 3:4].mT).expand(-1,(rays_d.shape[1]),-1)
    return rays_o, rays_d/torch.norm(rays_d,dim=-1,keepdim=True),torch.norm(rays_d,dim=-1,keepdim=True)

def strat_sampler(
        tn:torch.tensor,
        tf:torch.tensor,
        num_samples:int,
        exp:Optional [bool]=False,
        device:Optional[str]=None
)->torch.tensor:
    """Stratified sampling along a ray.
    Args:
        tn (torch.tensor): near depth
        tf (torch.tensor): far depth
        num_samples (int): number of samples
    Returns:
        t (torch.tensor): sampled points
    """
    # device=self.device
    if device is None:
        device='cuda' if torch.cuda.is_available() else 'cpu'
    
    if exp:
        t=torch.linspace(torch.log(tn),torch.log(tf),num_samples,device=device)
        t=t+(torch.rand_like(t)*(torch.log(tf)-torch.log(tn))/num_samples)
        t=torch.exp(t)
    else:
        t=torch.linspace(tn,tf,num_samples,device=device)
        t=t+(torch.rand_like(t)*(tf-tn)/num_samples)
    
    return t

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()

def cumprod_exclusive(
  tensor: torch.Tensor
) -> torch.Tensor:
  r"""
  (Courtesy of https://github.com/krrish94/nerf-pytorch)

  Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

  Args:
  tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
    is to be computed.
  Returns:
  cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
    tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
  """

  # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
  cumprod = torch.cumprod(tensor, -1)
  # "Roll" the elements along dimension 'dim' by 1 element.
  cumprod = torch.roll(cumprod, 1, -1)
  # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
  cumprod[..., 0] = 1.
  
  return cumprod

def eikonal_value(grads):

    # Compute the squared norm of the gradient tensor along the last dimension
    norm = torch.sqrt(torch.sum(grads ** 2, dim=-1))
    return norm
def eikonal_loss(norm):
    return torch.mean((norm-1)**2)

def calc_psnr(pred,target):
    mse=torch.mean((pred-target)**2)
    psnr=10*torch.log10(1./mse)
    return psnr