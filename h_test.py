import os
from datasets.s_coco_set import CustomDataset
from torchvision.transforms import transforms
import torch
import pandas as pd


# Define transformations
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create custom dataset and dataloader
root_folder = 'datasets/'
custom_dataset = CustomDataset(root_folder, transform=transform)
# print(custom_dataset.__getitem__(1)[1].shape)
dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=1, shuffle=True)

# Iterate through the dataloader

datas=[]
for inputs, annotations in dataloader:
    datas.append((inputs,annotations))

    # annotation test

    print(inputs.shape)

    print(len(annotations))

    for i in annotations:
        print("""
[ Annotations key list ]
    1. segmentation: 마스크 데이터
    2. area: segmentation에 대한 전체 넓이
    3. iscrowd: 군집화 분류
    4. image_id: annotation가 속한 image의 id값
    5. bbox: bounding box
    6. category_id: 분류 id
    7. id: annotation이 할당 받은 id
              
▼▼▼ 아래는 annotation에 대한 key 값과 각 예시 데이터 ▼▼▼""")
        print(list(i.keys()))
        for key, value in i.items():
            print(f'[ key: {key} ]')
            print(f'    value: {value}')
            print()
        break
    break


    if len(datas) >= 10:
        break