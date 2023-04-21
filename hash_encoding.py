import torch
import torch.nn as nn
import numpy as np

class HashEncoder(nn.Module):
    def __init__(self,N_max,N_min,L,E=0,T=2**14,F=2,dim=2,mu=None,sigma=None,device=None):
        super().__init__()
        if device is None:
            device='cuda' if torch.cuda.is_available() else 'cpu'
        self.device=device
        self.N_max = torch.tensor(N_max)
        self.N_min = torch.tensor(N_min)
        self.b=torch.exp((torch.log(self.N_max)-torch.log(self.N_min))/L)
        Embedding_list=[]
        self.L=L
        self.F=F
        self.T=T
        self.E=E
        self.dim=dim
        self.sigma=1 if sigma is None else (sigma).to(self.device)
        self.mu=0 if mu is None else mu
        # self.mu=torch.tensor(self.mu).to(self.device)
        # self.sigma=torch.tensor(self.sigma).to(self.device)
        for i in range(self.L):
            Embedding_list.append(nn.Embedding(T,F,sparse=True))
            nn.init.uniform_(Embedding_list[-1].weight,a=-1e-4,b=1e-4)
        self.Embedding_list=(nn.ModuleList(Embedding_list))
    
    def hash_func(self,x:np.ndarray,T:torch.tensor):
        pis=torch.tensor(np.array([1, 2654435761,805459861],dtype=np.int32),device=self.device)
        # pis=torch.tensor(np.array([1, 2654435761,805459861],dtype=np.int32))
        ndims=self.dim
        if self.dim==2:
            prod=x*pis[None,:ndims]
        if self.dim==3:
            prod=x*pis[None,None,:ndims]
        val=torch.bitwise_xor(prod[...,0],prod[...,1])
        if ndims==3:
            val=torch.bitwise_xor(val,prod[...,2])
        val=val%T
        return val
    
    def get_indices(self,x:torch.tensor):
        x_val=torch.stack([torch.floor(x),torch.ceil(x)],dim=-1)
        out=torch.zeros(x.shape[0],2**self.dim,self.dim,device=self.device)
        for a in range(2):
            for b in range(2):
                if self.dim<3:
                    # tensor=torch.stack([x_val[:,0,a],x_val[:,1,b]],dim=-1)
                    # print("Tensor_shape:",tensor.shape,out[:,a*2+b,:].shape)
                    out[:,a*2+b,:]=torch.stack([x_val[:,0,a],x_val[:,1,b]],dim=-1)
                    continue
                for c in range(2):
                    # out[:,a*2+b*2+c]=x_l[:,a]
                    # tensor=torch.stack([x_val[:,0,a],x_val[:,1,b],x_val[:,2,c]],dim=-1)
                    # print("Tensor_shape:",tensor.shape)
                    out[:,a*4+b*2+c,:]=torch.stack([x_val[:,0,a],x_val[:,1,b],x_val[:,2,c]],dim=-1)

        return out.long()
    
    def ninear_interpolation(self,x:torch.tensor,feature_vecs:torch.tensor):
        x_val=torch.stack([torch.floor(x),torch.ceil(x)],dim=-1)
        d=(x_val[...,1]-x_val[...,0])
        idxs=d!=0
        diff=(x-x_val[...,0])
        diff[idxs]=diff[idxs]/d[idxs]
        diff[~idxs]=0
        if self.dim==3:
            c00=feature_vecs[:,0]*(1-diff[...,0:1])+feature_vecs[:,4]*diff[...,0:1]
            c01=feature_vecs[:,1]*(1-diff[...,0:1])+feature_vecs[:,5]*diff[...,0:1]
            c10=feature_vecs[:,2]*(1-diff[...,0:1])+feature_vecs[:,6]*diff[...,0:1]
            c11=feature_vecs[:,3]*(1-diff[...,0:1])+feature_vecs[:,7]*diff[...,0:1]

            c0=c00*(1-diff[...,1:2])+c10*diff[...,1:2]
            c1=c01*(1-diff[...,1:2])+c11*diff[...,1:2]

            c=c0*(1-diff[...,2:3])+c1*diff[...,2:3]
        if self.dim==2:
            c0=feature_vecs[:,0]*(1-diff[...,0:1])+feature_vecs[:,1]*diff[...,0:1]
            c1=feature_vecs[:,2]*(1-diff[...,0:1])+feature_vecs[:,3]*diff[...,0:1]

            c=c0*(1-diff[...,1:2])+c1*diff[...,1:2]
        return c
    
    def forward(self,x,aux=None):
        assert(x.shape[-1]==self.dim)
        y=torch.zeros(x.shape[0],self.F*self.L+self.E,device=self.device)
        for i in range(self.L):
            N_l=self.N_min*self.b**i
            un_x=((x-self.mu)/self.sigma)*N_l
            # print(un_x.shape,x.shape,self.mu.shape,self.sigma.shape)
            idxs=self.get_indices(un_x)
            idx_hash=self.hash_func(idxs,T=self.T)
            interp_vec=self.ninear_interpolation(un_x,(self.Embedding_list[i])(idx_hash))
            y[:,i*self.F:(i+1)*self.F]=interp_vec
        return y