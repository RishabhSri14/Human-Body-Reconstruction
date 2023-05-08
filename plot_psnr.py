import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--pred_dirs',type=str,nargs='+',help='Give list of pred directories')
parser.add_argument('--gt_dirs',type=str,nargs='+',help='Give list of gt directories')
args=parser.parse_args()

def psnr(pred, gt,normalize=True):
    pred=pred.astype(np.float32)
    gt=gt.astype(np.float32)
    if normalize:
        pred=pred/255.0
        gt=gt/255.0
    mse=np.mean((pred-gt)**2)
    psnr=10*np.log10(1/mse)
    return psnr

def psnr_dir(pred_dir,gt_dir,normalize=True):
    pred_list=glob.glob(pred_dir+"/*.png")
    gt_list=glob.glob(gt_dir+"/*.png")
    pred_list.sort()
    gt=cv2.imread(gt_list[0])
    psnr_list=[]
    for pred_path in pred_list:
        pred=cv2.imread(pred_path)
        psnr_list.append(psnr(pred,gt,normalize))
    return np.array(psnr_list)

if __name__=='__main__':
    gt_dir=args.gt_dirs[0]
    min_len=1e7
    colors=['r','g','b','y','c','m','k']
    for pred_dir in args.pred_dirs:
        psnr_list=psnr_dir(pred_dir,gt_dir)
        if len(psnr_list)<min_len:
            min_len=len(psnr_list)
    for i,pred_dir in enumerate(args.pred_dirs):
        psnr_list=psnr_dir(pred_dir,gt_dir)
        # print(f"PSNR for {pred_dir} is {np.mean(psnr_list)}")
        x_axis=np.arange(0,min_len)*40
        plt.plot(x_axis,psnr_list[:min_len],'-o',label=pred_dir,color=colors[i])
        print(f"MEAN_PSNR for {pred_dir}:",psnr_list[-1])
    plt.title('PSNR vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.legend()
    plt.savefig('psnr.png')

