import os
import sys

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torchvision import transforms
from torchvision.models import resnet50
import torch.nn as nn
import torch.utils.data
from torch.optim import SGD
import torch.nn.functional as F

from dataset.s_coco_set import CustomDataset
import numpy as np

import rpn.nms as nms
import rpn.ss as ss
import roi_align.roi_align as roi_align
from mask.mask import Mask
from cls_bbox.cls_bbox import Classifier
import model as m


transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Create custom dataset and dataloader
root_folder = 'dataset/'
custom_dataset = CustomDataset(root_folder, transform=transform)
dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=1, shuffle=True)



def ch_box_shape(boxes):
    re_box = []
    for i in boxes:
        x, y, w, h = i
        re_box.append((int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)))
    return re_box

#====================[feature map]==============================

#====================[RoI]======================================

#====================[mask]=====================================

#====================[cls, bbox]================================

#====================[model]====================================
model = m.Mask_RCNN(num_classes=80)
optimizer  = SGD(model.parameters(), lr=0.1, momentum=0.9)

# Define optimizer and learning rate scheduler
# optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
# lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop===============================================================
num_epochs = 10
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
batch_size = 1
num_classes = 80
for epoch in range(num_epochs):
    datas=[]
    for inputs, annotations in dataloader:
        print("dsfdsfdsf",len(annotations))
        datas.append((inputs,annotations))
        # print("김성현",inputs)
        # print("김성현",inputs.shape)
        # print("최다예",annotations[0]['segmentation'])
        # print("최다예1",annotations[1])
        # print("최다예2",annotations[2])
        # print(annotations)
        # print(annotations['segmentation'])
        # print(annotations['bbox'])

        # ResNet-50 모델 불러오기
        resnet50_model = resnet50(pretrained=True)

        # 4번째 레이어까지의 모델 정의
        model_up_to_layer4 = torch.nn.Sequential(*list(resnet50_model.children())[:-2])

        # 모델에 이미지 전달하여 특성 맵 얻기
        with torch.no_grad():
            model_up_to_layer4.eval()
            features = model_up_to_layer4(inputs)
        img_ss = inputs.squeeze(0).numpy().transpose(1, 2, 0)
        rects = ss.selective_search(img_ss)
        boxes, idx = nms.nms(rects)
        roi_boxes = ch_box_shape(boxes)

        rois=[]
        total_loss = 0
        for tpl, val in zip(roi_boxes, idx):
            rois.append([val] + list(tpl))
        rois_tensor = torch.as_tensor(rois, dtype=torch.float)
            
        valid_rois = rois_tensor[:, 0] < features.size(0)
        filtered_rois_tensor = rois_tensor[valid_rois]
        spatial_scale = 1.0 / 8.0
        pooled_featrues = roi_align.roi_align(features, filtered_rois_tensor, spatial_scale)

        # mask
        
        ms = Mask(num_classes)
        mask_criterion = nn.BCELoss()
        mask_target = 0
        mask_loss = mask_criterion(ms, mask_target)
        

        #class
        mrcnn_class_logits, mrcnn_probs, mrcnn_bbox = Classifier(num_classes)
        classification_criterion = nn.CrossEntropyLoss()
        class_target =0
        classification_loss = classification_criterion(mrcnn_probs, class_target)
        
        #box
        regression_criterion = nn.SmoothL1Loss()
        target_bbox = 0
        regression_loss = regression_criterion(mrcnn_bbox, target_bbox)
        

        losses1 = sum(loss for loss in loss_dict1)
        losses2 = sum(loss for loss in loss_dict2)
        losses = sum(losses1, losses2)

        # Backward pass and optimization
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}')