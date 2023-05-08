from PIL import Image
from glob import glob
import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

with open("config.yaml","r") as f:
    config=yaml.load(f,Loader=yaml.FullLoader)


def get_bounding_box(images):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    boxes=np.zeros((len(images),4))
    for i in range(len(images)):
        output = model((transforms.ToTensor()(images[i])).unsqueeze(0))
        box = output[0]['boxes'].detach().cpu().numpy()
        boxes[i] = box[0]
        
    return boxes


image_path =glob(config["segmentation"]["input"]+"/*")

images = [Image.open(i).convert('RGB') for i in image_path]
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
transform_dlv3 = transforms.Compose([
        transforms.Resize((config["segmentation"]["h"], config["segmentation"]["w"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
input_tensor_dlv3=torch.stack(([transform_dlv3(i) for i in images]),dim=0)
    

# Load the pre-trained model from PyTorch Hub
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
with torch.no_grad():
    output = model(input_tensor_dlv3)['out']
    
mask = output.argmax(1).byte().cpu().numpy()
final_mask=np.zeros(input_tensor_dlv3.shape)

boxes=get_bounding_box(images)
print(boxes.shape)
for i in range(mask.shape[0]):
    original_tensor = input_tensor_dlv3[i].cpu().numpy()
    original_tensor = original_tensor*std.reshape([3,1,1])+mean.reshape([3,1,1])
    
    final_mask[i]=np.where(mask[i]==15,original_tensor,0)
    plt.imsave("./SegmentedImages/"+image_path[i].split("/")[-1],(np.transpose(final_mask[i],(1,2,0))*255).astype(np.uint8))


fig,axes=plt.subplots(len(images), 2, figsize=(20, 100))
for i  in range(len(images)):
    axes[i,0].imshow(images[i])
    axes[i,0].plot([boxes[i][0], boxes[i][2]], [boxes[i][1], boxes[i][1]], color='r', linewidth=2)
    axes[i,0].plot([boxes[i][0], boxes[i][0]], [boxes[i][1], boxes[i][3]], color='r', linewidth=2)
    axes[i,0].plot([boxes[i][0], boxes[i][2]], [boxes[i][3], boxes[i][3]], color='r', linewidth=2) 
    axes[i,0].plot([boxes[i][2], boxes[i][2]], [boxes[i][1], boxes[i][3]], color='r', linewidth=2)
    axes[i,0].axis('off')
    axes[i,1].imshow((np.transpose(final_mask[i],(1,2,0))*255).astype(np.uint8),cmap='gray')
    axes[i,1].axis('off')
plt.savefig("./SegmentedImages/SegmentationResults.png")
