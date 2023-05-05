import torch
import torch.nn as nn
import numpy as np
import time
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
        pis=torch.tensor(np.array([1, 2654435761,805459861],dtype=np.int32),device=self.device)
        self.register_buffer('pis',pis,persistent=False)
        if self.dim==2:
            self.pis=self.pis[None,:dim]
        if self.dim==3:
            self.pis=self.pis[None,None,:dim]
        for i in range(self.L):
            Embedding_list.append(nn.Embedding(T,F,sparse=False))
            nn.init.uniform_(Embedding_list[-1].weight,a=-1e-4,b=1e-4)
        self.Embedding_list=(nn.ModuleList(Embedding_list))
        num_vertices=2**self.dim
        neigs = np.arange(num_vertices, dtype=np.int64)
        dims = np.arange(self.dim, dtype=np.int64)
        bin_mask = torch.tensor((neigs[:,None] & (1 << dims[None,:]))==0, dtype=bool) # (neig, dim)
        print(bin_mask)
        self.register_buffer('bin_mask', bin_mask, persistent=False)

    @torch.no_grad()
    def hash_func(self,x:torch.tensor,T:torch.tensor):
        # pis=torch.tensor(np.array([1, 2654435761,805459861],dtype=np.int32))
        ndims=self.dim
        # if self.dim==2:
        #     prod=x*self.pis
        # if self.dim==3:
        #     prod=x*self.pis
        prod=x*self.pis[:ndims]
        val=torch.bitwise_xor(prod[...,0],prod[...,1])
        if ndims==3:
            val=torch.bitwise_xor(val,prod[...,2])
        val=val%T
        # val=torch.fmod(val,T) <- Crashes
        return val
    
    def get_indices(self,x:torch.tensor,out=None):
        x_val=torch.stack([torch.floor(x),torch.ceil(x)],dim=-1)
        if out is None:
            out=torch.zeros(x.shape[0],2**self.dim,self.dim,device=self.device)
        if self.dim==2:
            out[:,0,0]=x_val[:,0,0]
            out[:,0,1]=x_val[:,1,0]
            out[:,1,0]=x_val[:,0,0]
            out[:,1,1]=x_val[:,1,1]

            out[:,2,0]=x_val[:,0,1]
            out[:,2,1]=x_val[:,1,0]
            out[:,3,0]=x_val[:,0,1]
            out[:,3,1]=x_val[:,1,1]
        if self.dim==3:
            #a=0,b=0,c=0
            out[:,0,0],out[:,0,1],out[:,0,2]=x_val[:,0,0],x_val[:,1,0],x_val[:,2,0]
            #a=0,b=0,c=1
            out[:,1,0],out[:,1,1],out[:,1,2]=x_val[:,0,0],x_val[:,1,0],x_val[:,2,1]
            #a=0,b=1,c=0
            out[:,2,0],out[:,2,1],out[:,2,2]=x_val[:,0,0],x_val[:,1,1],x_val[:,2,0]
            #a=0,b=1,c=1
            out[:,3,0],out[:,3,1],out[:,3,2]=x_val[:,0,0],x_val[:,1,1],x_val[:,2,1]
            #a=1,b=0,c=0
            out[:,4,0],out[:,4,1],out[:,4,2]=x_val[:,0,1],x_val[:,1,0],x_val[:,2,0]
            #a=1,b=0,c=1
            out[:,5,0],out[:,5,1],out[:,5,2]=x_val[:,0,1],x_val[:,1,0],x_val[:,2,1]
            #a=1,b=1,c=0
            out[:,6,0],out[:,6,1],out[:,6,2]=x_val[:,0,1],x_val[:,1,1],x_val[:,2,0]
            #a=1,b=1,c=1
            out[:,7,0],out[:,7,1],out[:,7,2]=x_val[:,0,1],x_val[:,1,1],x_val[:,2,1]

        # for a in range(2):
        #     for b in range(2):
        #         if self.dim<3:
        #             # tensor=torch.stack([x_val[:,0,a],x_val[:,1,b]],dim=-1)
        #             # print("Tensor_shape:",tensor.shape,out[:,a*2+b,:].shape)
        #             out[:,a*2+b,:]=torch.stack([x_val[:,0,a],x_val[:,1,b]],dim=-1)
        #             continue
        #         for c in range(2):
        #             # out[:,a*2+b*2+c]=x_l[:,a]
        #             # tensor=torch.stack([x_val[:,0,a],x_val[:,1,b],x_val[:,2,c]],dim=-1)
        #             # print("Tensor_shape:",tensor.shape)
        #             out[:,a*4+b*2+c,:]=torch.stack([x_val[:,0,a],x_val[:,1,b],x_val[:,2,c]],dim=-1)

        return out.long()
    
    def ninear_interpolation(self,x:torch.tensor,feature_vecs:torch.tensor):
        x_val=torch.stack([torch.floor(x),torch.ceil(x)],dim=-1).to(torch.long)
        # d=(x_val[...,1]-x_val[...,0])
        # idxs=d!=0
        diff=(x-x_val[...,0])
        # diff[idxs]=diff[idxs]
        # diff[~idxs]=0
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
    def fast_get_indices(self,x:torch.tensor,x_val:torch.tensor,out:torch.tensor=None):

        # if out is None:
        #     out=torch.zeros(x.shape[0],2**self.dim,self.dim,device=self.device)
        # idxs=torch.zeros_like(self.bin_mask,dtype=torch.long,device=self.device)
        # out[...,self.bin_mask]=x_val[...,0:1][self.bin_mask]
        # out[...,~(self.bin_mask)]=x_val[...,1:2][self.bin_mask]
        # print("X_VAL_SHAPE:",x_val.shape)
        out=torch.where(self.bin_mask,x_val[...,0],x_val[...,1])
        return out
    
    def fast_nlinear_interpolation(self,x:torch.tensor,feature_vecs:torch.tensor,x_val:torch.tensor,diff:torch.tensor):
        # wts=torch.zeros_like(feature_vecs,device=self.device)
        # wts[...,self.bin_mask]=1-diff
        # wts[...,~(self.bin_mask)]=diff
        wts=torch.where(self.bin_mask,1-diff,diff)
        wts=wts.prod(dim=-1,keepdim=True)
        return (feature_vecs*wts).sum(dim=-2)
    
    def forward(self,x,aux=None):
        assert(x.shape[-1]==self.dim)
        y=torch.zeros(x.shape[0],self.F*self.L+self.E,device=self.device)
        out=torch.zeros(x.shape[0],2**self.dim,self.dim,device=self.device)
        # t1_sum,t2_sum,t3_sum=0,0,0
        self.bin_mask=self.bin_mask.reshape((1,)*(len(x.shape)-1)+(2**self.dim,self.dim))
        for i in range(self.L):
            N_l=self.N_min*self.b**i
            un_x=((x-self.mu)/self.sigma)*N_l
            # print(un_x.shape,x.shape,self.mu.shape,self.sigma.shape)
            # t1=time.time()
            bdims = len(x.shape[:-1])
            x_val=torch.stack([un_x.long(),un_x.long()+1],dim=-1)
            diff=(un_x-x_val[...,0])
            x_val=x_val[...,None,:,:]
            diff=diff[...,None,:].detach()
            idxs=self.fast_get_indices(x,x_val,out)
            hash_idxs=self.hash_func(idxs,self.T)
            feature_vecs=self.Embedding_list[i](hash_idxs)
            interp_vecs=self.fast_nlinear_interpolation(x,feature_vecs,x_val,diff)
            y[:,i*self.F:(i+1)*self.F]=interp_vecs
            # t1_sum+=t1
            # t2_sum+=t2
            # t3_sum+=t3
        # print("Time:",t1_sum,t2_sum,t3_sum)
        return y