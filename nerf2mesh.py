import trimesh
import numpy as np
import matplotlib.pyplot as plt
import cv2
from encoder import *
from hash_encoding import *
from tqdm import tqdm
from helper import *
import argparse
import trimesh
import open3d as o3d
import os
from torchmcubes import marching_cubes, grid_interp

parser = argparse.ArgumentParser(description='Train Hashing')
parser.add_argument('--use_sdf',action='store_true',help='Use sdf formulation while training')
parser.add_argument('--hierarchical',action='store_true',help='Use hierarchical sampling')
parser.add_argument('--max_res',type=float,default=2048,help='Max resolution of the grid')
parser.add_argument('--hash_size',type=float,default=16,help='Log Size of the hash table')
parser.add_argument("--model_name",type=str,default='default',help='Name of saved model')
parser.add_argument("--bound_pth",type=str,default='bounds.npy',help='Path to bounds.npy')
parser.add_argument("--ckpt_name",type=str,default='N_2048_T_16',help='Name of checkpoint')
parser.add_argument('--near',type=float,default=2.0,help='Near point')
parser.add_argument('--far',type=float,default=6.0,help='Far point')

args = parser.parse_args()
resolution=256
bounds=np.load(args.bound_pth)
min_bound,max_bound=bounds[0],bounds[1]
print(min_bound,max_bound)
x=np.linspace(min_bound[0],max_bound[0],resolution)
y=np.linspace(min_bound[1],max_bound[1],resolution)
z=np.linspace(min_bound[2],max_bound[2],resolution)

min_bound=torch.tensor(min_bound)
max_bound=torch.tensor(max_bound)
X,Y,Z=np.meshgrid(x,y,z)
X,Y,Z=torch.tensor(X.reshape(-1)),torch.tensor(Y.reshape(-1)),torch.tensor(Z.reshape(-1))
grid=torch.stack([X,Y,Z],dim=1)
grid=grid.to(torch.float16)
print(grid.shape)

device='cuda' if torch.cuda.is_available() else 'cpu'
L=16
F=2
num_freq=4
N_max=args.max_res
T=int(2**args.hash_size)
# max_bound,min_bound=find_bounding_box2(train_loader,near=torch.as_tensor(2.0),far=torch.as_tensor(6.0),K=K)
near=torch.tensor(args.near)
far=torch.tensor(args.far)
print("BOUNDING BOX:",max_bound,min_bound)
mu=(min_bound).to(device)

sigma=((max_bound-min_bound)**2).sum().sqrt().to(device)
encoder=HashEncoder(N_min=16,N_max=N_max,L=L,F=F,T=T,E=0,dim=3,mu=mu,sigma=sigma)
nerf=torch.nn.DataParallel(MLP_3D(num_sig=2,num_col=2,L=L,F=F,d_view=3*num_freq*2,max_bound=max_bound,min_bound=min_bound))
dir_encoder=PositionalEncoder(d_model=3,num_freq=num_freq)
nerf_ckpt=args.ckpt_name+'_Nerf_hash.pth'
encoder_ckpt=args.ckpt_name+'_encoder_hash.pth'
nerf.load_state_dict(torch.load(nerf_ckpt))
encoder.load_state_dict(torch.load(encoder_ckpt))
nerf=nerf.to(device)
encoder=encoder.to(device)
dir_encoder=dir_encoder.to(device)

print("GRID_SHAPE:",grid.shape)
print("GRID_DEVICE:",grid.device)
view_dirs=torch.zeros_like(grid,dtype=torch.float16)
view_dirs[...,2]=1.0
dataload=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(grid,view_dirs),batch_size=400000,shuffle=False,num_workers=8,pin_memory=True)
if not os.path.exists('density_grid_w_rgb.npy'):
    out=[]
    with torch.no_grad():
        for batch in tqdm(dataload):
            samples,view_dir=batch
            samples=samples.to(device)
            view_dir=view_dir.to(device)
            print(samples.shape)
            samples=encoder(samples)
            view_dir=dir_encoder(view_dir)
            output=nerf(samples,view_dir)
            print(output.shape)
            out.append(output.cpu().numpy())
    out_grid=np.concatenate(out,axis=0)
    out_grid=out_grid.reshape(resolution,resolution,resolution,4)
    np.save('density_grid_w_rgb.npy',out_grid)
out_grid=np.load('density_grid_w_rgb.npy')
torch.cuda.empty_cache()
out_grid=torch.tensor(out_grid)
# color_grid=out_grid[...,:3]
color_grid=torch.stack((X,Y,Z),axis=-1)
color_grid=color_grid.reshape(resolution,resolution,resolution,3).to(torch.float32)
color_grid=(color_grid-color_grid.min())/(color_grid.max()-color_grid.min())
density_grid=out_grid[...,-1].clone()
print("DENSITY_GRID_SHAPE:",density_grid.shape,color_grid.shape)
color_grid=torch.permute(color_grid,(3,2,1,0)).clone().contiguous()
verts, faces = marching_cubes(density_grid, 30.0)
colrs = grid_interp(color_grid, verts)
print("MIN_MAX COLORS:",colrs.max, colrs.min)
# density_grid=np.random.rand(512,512,512)
mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(verts)
mesh.triangles = o3d.utility.Vector3iVector(faces)
mesh.vertex_colors = o3d.utility.Vector3dVector(colrs)
wire = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
o3d.visualization.draw_geometries([mesh, wire], window_name='Marching cubes (CUDA)')


print(density_grid.shape)
# Create Open3D voxel grid