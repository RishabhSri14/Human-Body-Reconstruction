import torch
from hash_encoding import HashEncoder
import cv2
from tqdm import tqdm
import numpy as np

class MLP_2D(torch.nn.Module):
    def __init__(self,L=16,F=2,E=0):
        super().__init__()
        self.linear1=torch.nn.Linear(L*F+E,64)
        self.relu=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(64,3)
    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        x=self.relu(x)
        return x

class MLP_3D(torch.nn.Module):
    def __init__(self,num_sig=3,num_col=2,h_size=64,d_view=3,L=16,F=2,E=0):
        super().__init__()
        self.d_view=d_view
        sig_list=[]
        sig_list.append(torch.nn.Linear(L*F+E,h_size))
        sig_list.append(torch.nn.ReLU())
        for i in range(num_sig):
            if i==num_sig-1:
                sig_list.append(torch.nn.Linear(h_size,1+15))
            else:
                sig_list.append(torch.nn.Linear(h_size,h_size))
                sig_list.append(torch.nn.ReLU())
        self.sigmoid=torch.nn.Sigmoid()
        self.sig_model=torch.nn.Sequential(*sig_list)
        self.relu=torch.nn.ReLU()
        self.elu=torch.nn.ELU()
        self.lrelu=torch.nn.LeakyReLU()

        col_list=[]
        col_list.append(torch.nn.Linear(15+d_view,h_size))
        col_list.append(torch.nn.ReLU())
        for i in range(num_col):
            if i==num_col-1:
                col_list.append(torch.nn.Linear(h_size,3))
            else:
                col_list.append(torch.nn.Linear(h_size,h_size))
                col_list.append(torch.nn.ReLU())
        self.col_model=torch.nn.Sequential(*col_list)

    def forward(self,x,viewdirs=None,mask=None):
        dens_vec=self.sig_model(x)
        density=dens_vec[:,0:1]
        # print("SHAPES",dens_vec.shape,density.shape,x.shape)
        # print("DENSITY_MIN_MAX:",density[:5],density.min(dim=0),density.max(dim=0))
        if density.max()<0:
            print("WARNING: DENSITY IS NEGATIVE!")
        density=self.lrelu(density)
        
        feat_vec=dens_vec[:,1:]
        if viewdirs is not None:
            rgb=self.col_model(torch.concat((feat_vec,viewdirs),dim=-1))            
            rgb=self.elu(rgb)
            out= torch.concat((rgb,density),dim=-1) # Output format: (RGB,sigma)
            if mask is not None:
                out=out*mask[...,None]
            return out
        else:
            if mask is not None:
                return density*mask
            else:
                return density

def train(model,num_epoch,train_loader,test_loader,encoder,shape,device='cuda',display=False,display_write=False):
    optimizer_embed=torch.optim.SparseAdam(list(encoder.Embedding_list.parameters()),lr=0.01)
    optimizer_MLP=torch.optim.AdamW(model.parameters(),lr=0.01)
    criterion=torch.nn.MSELoss()
    
    scheduler_embed = torch.optim.lr_scheduler.OneCycleLR(optimizer_embed, 
                       max_lr = 5e-2, # Upper learning rate boundaries in the cycle for each parameter group
                       steps_per_epoch = len(train_loader), # The number of steps per epoch to train for.
                       epochs = num_epoch, # The number of epochs to train for.
                       anneal_strategy = 'cos') 
    scheduler_MLP = torch.optim.lr_scheduler.OneCycleLR(optimizer_MLP, 
                       max_lr = 5e-2, # Upper learning rate boundaries in the cycle for each parameter group
                       steps_per_epoch = len(train_loader), # The number of steps per epoch to train for.
                       epochs = num_epoch, # The number of epochs to train for.
                       anneal_strategy = 'cos') 
    model.train()
    img=torch.zeros(shape,dtype=torch.uint8,device=device)
    for epoch in range(num_epoch):
        pbar=tqdm(enumerate(train_loader),total=len(train_loader))
        for i,(x,y) in pbar:
            x=x.to(device)
            y=y.to(device)
            optimizer_MLP.zero_grad()
            optimizer_embed.zero_grad()
            
            x_enc=encoder(x)
            # print(x.shape,y.shape)
            y_pred=model(x_enc)

            # print(y_pred.shape)
            loss=criterion(y_pred,y)
            loss.backward()
            optimizer_MLP.step()
            optimizer_embed.step()
            scheduler_embed.step()
            scheduler_MLP.step()
            pbar.desc=f"Epoch: {epoch} loss: {loss}"

            if display_write==True and epoch%50==0 and i==0:
                with torch.no_grad():
                    for i,(x,y) in tqdm(enumerate(test_loader),total=len(test_loader)):
                        x=x.to(device)
                        y=y.to(device)
                        x=encoder(x)
                        if i==0:
                            y_pred=model(x)
                        else:
                            y_pred=torch.cat([y_pred,model(x)],dim=0)
                        # loss=criterion(y_pred,y)
                    y_pred_np=y_pred.cpu().detach().numpy()
                    y_pred_np=y_pred_np.reshape(shape)
                    cv2.imshow('pred',y_pred_np)
                    cv2.imwrite(f'pred{epoch}.jpg',y_pred_np*255.0)
                    key=cv2.waitKey(1)
            x1=torch.round((x[:,0])).long()
            x2=torch.round((x[:,1])).long()
            img[x2,x1,:]=(y_pred*255).to(torch.uint8)
            
            img_np=img.detach().cpu().numpy()

            cv2.imshow('output',img_np)
            key=cv2.waitKey(1)
            if key==ord('q'):
                exit(0)
                # print(loss.item())
                # pbar.desc(f"loss: {loss}")
                    # break
    cv2.destroyAllWindows()

if __name__=="__main__":
    torch.autograd.set_detect_anomaly(True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    L=16
    F=2
    model=torch.nn.DataParallel(MLP_2D(L=L,F=F))
    img=cv2.imread("mountain.png")
    img_shape=np.array((img.shape[1],img.shape[0]))
    encoder=(HashEncoder(N_min=16,N_max=2**16,L=L,F=F,T=2**18,sigma=img_shape))
    encoder.to(device)
    model.to(device)
    optimizer_embed=torch.optim.SparseAdam(list(encoder.Embedding_list.parameters()),lr=0.2)
    optimizer_MLP=torch.optim.AdamW(model.parameters(),lr=0.2)
    # img=cv2.imread("mountain.png")
    print(img.shape)

    shape=img.shape
    i,j=torch.meshgrid(torch.arange(0,img.shape[1]),torch.arange(0,img.shape[0]),indexing='xy')
    i,j=i.flatten(),j.flatten()
    print(i[:10],j[:10])

    x=torch.stack([i,j],dim=-1).float()
    y_gt=torch.from_numpy(img).float().reshape(-1,3)/255.0
    print(x.shape,y_gt.shape)
    train_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y_gt),batch_size=200000,shuffle=True,num_workers=4,pin_memory=True)
    test_loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x,y_gt),batch_size=200000,shuffle=False,num_workers=4,pin_memory=True)

    train(model,100,train_loader,test_loader,encoder,shape,'cuda')
    torch.save(model.state_dict(),'model_hash.pth')
