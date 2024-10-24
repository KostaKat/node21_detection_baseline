import os
import numpy as np
import torch
from PIL import Image
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset  # Explicitly inherit from PyTorch Dataset
import training_utils.transforms as T
from torchvision import transforms as torch_transforms
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    # Add the normalization step to the pipeline
    return T.Compose(transforms)



class CXRNoduleDataset(Dataset):
    def __init__(self, root, csv_file, transforms):
        self.root = root
        self.transforms = transforms
        self.data = pd.read_csv(csv_file)
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.imgs = [i for i in self.imgs if i in self.data['img_name'].values]
        # Read only image files in following format
        self.imgs = [i  for i in self.imgs if os.path.splitext(i)[1].lower() in (".mhd", ".mha", ".dcm", ".png", ".jpg", ".jpeg")]   
     
    def __getitem__(self, idx):
        
        img_path = os.path.join(self.root, "images", str(self.imgs[idx]))
        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))

        img_array = np.asarray(img)
        img =  (img_array / 65535.0 * 255).astype(np.uint8)

        # Convert to mode "L" for 8-bit grayscale
        img = Image.fromarray(img)
        
        nodule_data = self.data[self.data['img_name']==str(self.imgs[idx])]
        num_objs = len(nodule_data)

        boxes = []
        
        if nodule_data['label'].any()==1: # nodule data
            for i in range(num_objs):
                x_min = int(nodule_data.iloc[i]['x'])
                y_min = int(nodule_data.iloc[i]['y'])
                y_max = int(y_min+nodule_data.iloc[i]['height'])
                x_max = int(x_min+nodule_data.iloc[i]['width'])
                boxes.append([x_min, y_min, x_max, y_max])

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            labels = torch.ones((num_objs,), dtype=torch.int64)
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        # for non-nodule images
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            area =  torch.tensor([])
            labels = torch.zeros((0,), dtype=torch.int64)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

            
        image_id = torch.tensor([idx])
        # suppose all instances are not crowd
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            #normalize the image
        
        image_name = str(self.imgs[idx])

        return img, target, image_name

    def __len__(self):
        return len(self.imgs)
    def get_labels(self):
        labels = []

        for img_name in self.imgs:
            # Get all labels in the CSV file for this image
            nodule_data = self.data[self.data['img_name'] == img_name]['label'].values

            # If there's any label of 1, classify this image as a nodule (label 1), otherwise non-nodule (label 0)
            if 1 in nodule_data:
                labels.append(1)
            else:
                labels.append(0)

        return labels

        
    
    
    
    