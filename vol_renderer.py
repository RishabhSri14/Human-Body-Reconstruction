import torch
from typing import Tuple,Optional
import numpy as np
import h5py as h5
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from encoder import *

class NeRF(nn.Module):
  r"""
  Neural radiance fields module.
  """
  def __init__(
    self,
    d_input: int = 3,
    n_layers: int = 8,
    d_filter: int = 256,
    skip: Tuple[int] = (4,),
    d_viewdirs: Optional [int] = None
  ):
    super().__init__()
    self.d_input = d_input
    self.skip = skip
    self.act = nn.ReLU()
    self.d_viewdirs = d_viewdirs
    self.sigmoid = nn.Sigmoid()
    # Create model layers
    self.layers = nn.ModuleList(
      [nn.Linear(self.d_input, d_filter)] +
      [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
       else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
    )

    # Bottleneck layers
    if self.d_viewdirs is not None:
        # If using viewdirs, split alpha and RGB
        self.alpha_out = nn.Linear(d_filter, 1)
        self.rgb_filters = nn.Linear(d_filter, d_filter)
        self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
        self.output = nn.Linear(d_filter // 2, 3)
    else:
        # If no viewdirs, use simpler output
        self.output = nn.Linear(d_filter, 4)
  
  def forward(
    self,
    x: torch.Tensor,
    viewdirs: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
    r"""
    Forward pass with optional view direction.
    """

    # Cannot use viewdirs if instantiated with d_viewdirs = None
    if self.d_viewdirs is None and viewdirs is not None:
        raise ValueError('Cannot input x_direction if d_viewdirs was not given.')

    # Apply forward pass up to bottleneck
    x_input = x
    for i, layer in enumerate(self.layers):
        x = self.act(layer(x))
        if i in self.skip:
            x = torch.cat([x, x_input], dim=-1)
    #   print(f"{i}_layer_in",x.shape)

    # Apply bottleneck
    if self.d_viewdirs is not None:
        # Split alpha from network output
        alpha = self.alpha_out(x)
        alpha = self.sigmoid(alpha)
        # Pass through bottleneck to get RGB
        x = self.rgb_filters(x)
        x = torch.concat([x, viewdirs], dim=-1)
        x = self.act(self.branch(x))
        x = self.output(x)
        
        x = self.act(x)
        # Concatenate alphas to output
        x = torch.concat([x, alpha], dim=-1)
    else:
        # Simple output
        x = self.output(x)
    return x


def get_od(
    H:int,
    W:int,
    K:torch.tensor,
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
    if c2w.get_device()>=0:
        device="cuda"
    else:
        device="cpu"

    # device="cuda" if torch.cuda.is_available() else "cpu"
    i, j = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    # Get direction of rays, going theough a pixel, first calcuate ICW->CCW
    # K=[[fx, 0, cx],
    #    [0, fy, cy], 
    #    [0, 0, 1]]
    i,j=i.to(device),j.to(device)
    # i1,j1=i.transpose(-1,-2),j.transpose(-1,-2)
    i = ((i - K[0,2]) / K[0,0] ).reshape(-1)
    j = ((j - K[1,2]) / K[1,1] ).reshape(-1)
    # transform to camera coordinates: ICW->CCW (only direction matters)
    dirs = torch.stack((i, -j, -torch.ones_like(i)), axis=-1)
    # rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    # rays_d=(c2w[:3,:3]@(dirs.T)).T
    if find_inv:
        mat=c2w[:3,:3]
        rays_d=(torch.linalg.inv(c2w[...,:3,:3])@dirs.mT).mT
    else:
        # print("DIRS_SHAPE:",dirs.shape)
        rays_d=((c2w[...,:3,:3])@(dirs.mT)).mT
    rays_o = (c2w[...,:3, 3:4].mT).expand(-1,(rays_d.shape[1]),-1)
    # rays_o = (c2w[:3, 3:4].mT).expand(rays_d.shape)
    # transform to world coordinates
    # rays_d = torch.einsum('...ij,...j->...i', K, rays_d)
    # rays_o = torch.einsum('...ij,...j->...i', K, rays_o)
    # print("SHAPES:",rays_o.shape,rays_d.shape)
    return rays_o, rays_d/torch.norm(rays_d,dim=-1,keepdim=True)
    # return rays_o, rays_d


def strat_sampler(
        tn:torch.tensor,
        tf:torch.tensor,
        num_samples:int,
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
    if device is None:
        device="cuda" if torch.cuda.is_available() else "cpu"

    t=torch.linspace(tn,tf,num_samples,device=device)
    t=t+(torch.rand_like(t)*(tf-tn)/num_samples)
    return t

# def model(rays:torch.tensor)->torch.tensor:
#     """NeRF model.
#     Args:
#         rays (torch.tensor): rays, (H*W, 3)
#     Returns:
#         model_out (torch.tensor): model output, (H*W, 4)
#     """
#     device="cuda" if torch.cuda.is_available() else "cpu"

#     f = h5.File('datacube.hdf5', 'r')
#     datacube = np.array(f['density'])
#     datacube=torch.from_numpy(datacube).to(device)
#     # print("t unique",torch.unique(rays.mean(dim=-1)))
#     model_out=torch.ones((rays.shape[0],rays.shape[1],4),device=device)
#     x=datacube[rays[...,0].long(),rays[...,1].long(),rays[...,2].long()]
#     r = 1.0*torch.exp( -(x - 9.0)**2/1.0 ) +  0.1*torch.exp( -(x - 3.0)**2/0.1 ) +  0.1*torch.exp( -(x - -3.0)**2/0.5 )
#     g = 1.0*torch.exp( -(x - 9.0)**2/1.0 ) +  1.0*torch.exp( -(x - 3.0)**2/0.1 ) +  0.1*torch.exp( -(x - -3.0)**2/0.5 )
#     b = 0.1*torch.exp( -(x - 9.0)**2/1.0 ) +  0.1*torch.exp( -(x - 3.0)**2/0.1 ) +  1.0*torch.exp( -(x - -3.0)**2/0.5 )
#     model_out[...,3]=x
#     model_out[...,:3]=torch.stack((r,g,b),dim=-1)
#     return model_out



def vol_render(
        model:NeRF,
        rays_d:torch.Tensor,
        rays_o:torch.Tensor,
        near:float=0.0,
        far:float=1.0,
        num_samples=100,
        Pos_encode:Optional [PositionalEncoder]=None,
        Dir_encode:Optional [PositionalEncoder]=None,
        t:Optional [torch.Tensor]=None,
    )->Tuple[torch.Tensor,torch.Tensor]:
    # device="cuda" if torch.cuda.is_available() else "cpu"

    if rays_d.get_device()>=0:
        device="cuda"
    else:
        device="cpu"
    # print("DEVICE:!!",device)
    if t is None:
        t=strat_sampler(near,far,num_samples,device=device)
        # t=t[None,:,None]
    rays=rays_o[...,None,:]+rays_d[...,None,:]*t[None,:,None]
    # print("RAYS_SHAPE:",rays[...,0].max(),rays[...,1].max(),rays[...,2].max(),rays[...,0].min(),rays[...,1].min(),rays[...,2].min())
    orig_shape=rays.shape
    # print("t_shape:",rays_o[:,None,:].shape)
    if Pos_encode is not None and Dir_encode is not None:
        rays=rays.reshape(-1,3)
        dirs=rays_d[...,None,:].repeat(1,num_samples,1)
        dirs=dirs.reshape(-1,3)
        rays=Pos_encode(rays)
        dirs=Dir_encode(dirs)
        rays=rays.reshape(rays.shape[0],-1)
        dirs=dirs.reshape(dirs.shape[0],-1)
    elif Pos_encode is not None and Dir_encode is None:
        rays=rays.reshape(-1,3)
        dirs=rays_d[...,None,:].repeat(1,num_samples,1)
        dirs=dirs.reshape(-1,3)
        rays=Pos_encode(rays)
    else:
        print("ERROR: No positional encoding")
        # return
    # else:
    #     print("SOMEHOW HERE!!")
    model_out=model(rays,dirs)
    model_out=model_out.reshape(orig_shape[0],orig_shape[1],-1)
    sigma=model_out[...,3]
    rgb=model_out[...,0:3]

    del_t=torch.zeros_like(t).to(device)

    del_t[...,:-1]=t[...,1:]-t[...,:-1]
    # del_t[...,-1]=1e10
    del_t=del_t[None,:]

    prod=sigma*del_t
    T=torch.exp(-torch.cumsum(prod,axis=-1))
    alpha=1-torch.exp(-torch.nn.functional.relu(prod+torch.randn_like(prod)*0.001))
    C=torch.sum(T[:,:,None]*alpha[:,:,None]*rgb,dim=-2)

    return C

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

def find_bounding_box(data_loader,near,far,K,device=None):
    if device is None:
        device=K.device
    # K[0,0]=focal
    # K[1,1]=focal
    W=2*K[0,2]
    H=2*K[1,2]
    # K=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]])).to(device)
    with torch.no_grad():
        for batch in data_loader:
            # H,W=image.shape[:2]
            img,pose,_=batch
            pose=pose.to(device)
            rays_o, rays_d = get_od(H, W, K, pose)
            t=strat_sampler(near,far,12,device=device)
            rays=rays_o[...,None,:]+rays_d[...,None,:]*t[None,:,None]
            rays=rays.reshape(-1,3)
            min_bound=torch.ones(3,device=device)*(1e7)
            max_bound=torch.ones(3,device=device)*(-1e7)

            for i in range(3):
                min_elem=torch.min(rays[:,i])
                max_elem=torch.max(rays[:,i])
                if min_bound[i]>min_elem:
                    min_bound[i]=min_elem
                if max_bound[i]<max_elem:
                    max_bound[i]=max_elem
    return max_bound,min_bound

######################################################################
def make_batch(
        in_rays:torch.Tensor,
        batch_size: int,
    )->list:
    batches=[]
    for i in range(0,in_rays.shape[0],batch_size):
        batches.append(in_rays[i:i+batch_size].to('cpu'))
    return batches

if __name__=="__main__":
    # print(datacube.shape)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print("deivce:",device)
    # H=256
    # W=256

    K=torch.from_numpy(np.array([[1,0,0],[0,1,0],[0,0,1]])).to(device)
    # c2w=torch.randint(5,(4,4)).to(torch.float32).to(device)
    num_freq=10
    model=NeRF(d_input=3*num_freq*2,d_viewdirs=2*num_freq*2)
    model=model.to(device)

    data = np.load('tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    train_imgs=images[:-1,...]
    test_imgs=images[-1:-2:-1,...]
    train_pose=poses[:-1,...]
    test_pose=poses[-1:-2:-1,...]
    H,W,_=images.shape[1:]
    # print(images.shape,train_imgs.shape,test_imgs.shape)

    K[0,0]=torch.tensor(focal)
    K[1,1]=torch.tensor(focal)
    K[0,2]=W/2
    K[1,2]=H/2
    print("CAMERA SHAPES:",images.shape,poses.shape,focal)
    c2w=torch.from_numpy(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dtype=np.float32)).to(device)
    c2w=torch.tensor(poses[0,...],device=device)
    print("c2w:",c2w)
    rays_o, rays_d=get_od(H,W,K,c2w)
    origins=rays_o.cpu().numpy()
    dirs=rays_d.cpu().numpy()

    print("MIN_MAX:",rays_d.min(),rays_d.max())
    C=vol_render(rays_o,rays_d,num_samples=100,far=50)
    print(C[0])
    C=C.reshape(H,W,3).cpu().numpy()
    plt.imshow(C)

    plt.show()
