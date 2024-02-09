import os
from torchvision import transforms
from torch.utils.data import Dataset
import torch
from PIL import Image
import json
# from pycocotools.coco import COCO
# import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.image_folder = os.path.join(root_folder, 'images/val2017')
        self.annotation_folder = os.path.join(root_folder, 'annotations/')

        annotation_name = os.path.join(self.annotation_folder, "instances_val2017.json")
        # Parse JSON annotation
        with open(annotation_name, 'r') as f:
            ann_info = json.load(f)
        if ann_info:
            # print(ann_info.keys())
            self.categories=ann_info['categories']
            self.image_info=ann_info['images']
            self.annotations=ann_info['annotations']
            # print(self.categories)


        self.image_list = os.listdir(self.image_folder)
        self.annotation_list = [f.replace('.jpg', '.json') for f in self.image_list]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
                
        file_name=self.image_info[idx]['file_name']
        image_id=self.image_info[idx]['id']


        img_name = os.path.join(self.image_folder, file_name)
        print(img_name)

        image = Image.open(img_name).convert('RGB')
        ann_list = []
        for annotation in self.annotations:
            if annotation['image_id'] == image_id:
                ann_list.append(annotation)
        print(ann_list)

        # Your parsing logic here based on the structure of your JSON annotations

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, ann_list
    
def get_label_list():
    label_list=['background']

    with open("datasets/coco-labels-2014_2017.txt", "r") as f:
        for line in f:
            label_list.append(line)
    
    return label_list

# # Define transformations
# transform = transforms.Compose([
#     transforms.Resize((1024, 1024)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# # Create custom dataset and dataloader
# root_folder = 'datasets/'
# custom_dataset = CustomDataset(root_folder, transform=transform)
# # print(custom_dataset.__getitem__(1)[1].shape)
# dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=1, shuffle=True)

# # Iterate through the dataloader

# datas=[]
# for inputs, annotations in dataloader:
#     # Your processing logic here
#     # print("hear----------------------")
#     # print(inputs.shape,len(annotations))
#     datas.append((inputs,annotations))
#     # if len(datas) >= 10:
#     break
