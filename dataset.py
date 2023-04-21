from glob import glob
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import torch
import cv2
import torchvision.transforms as T
import json
class NeRF_DATA(Dataset):
    def __init__(self,
                 json_path,
                 transforms=None):

        super.__init__()
        assert os.path.exists(json_path), "The path {} does not exist".format(json_path)
        self.path=json_path
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.camera_angle_x = self.data['camera_angle_x']
        self.dataset = self.data['frames']
        self.image_transforms = transforms
        # print('\n\n', self.image_transforms, '\n')

   
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filename = self.path[:self.path.rfind('/')]+self.dataset[idx]['file_path'][self.dataset[idx]['file_path'].find('.')+1:]+'.png'
        assert os.path.exists(filename) , "The file {} does not exist".format(filename)
        image =cv2.imread(filename)
        image  = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.image_transforms:
            image = self.image_transforms(image, return_tensors="pt")
        else:
            image = T.ToTensor()(image)
            
        return image,torch.Tensor(self.dataset[idx]['transform_matrix']),self.dataset[idx]['rotation']
