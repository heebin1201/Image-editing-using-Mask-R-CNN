import torch

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
import pandas as pd
import glob
import PIL
from PIL import Image, ImageOps
import skimage
from skimage import draw
import h5py
import torch
from torch.autograd import Variable
import torch.utils.data
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms
# import pycocotools

## image test code ##
# image = skimage.io.imread(os.path.join('/Users/hyeok/Desktop/Development/Python/osop/test_out/ex_mask/images/2502287818_41e4b0c4fb_z.jpg'))
# print(image.shape)


class Test_out:

    def __init__(self, image_path):
        self.TEST_INPUT_IMAGE_PATH = image_path
        self.input_image=self.get_image()
        self.feature_map=None

    ## 이미지를 불러오고 전처리 된 이미지 리턴 ##
    # Input : 이미지
    # shape : (batch_size, channels, height, width)
    def get_image(self):
        img_tensor=None
        img = Image.open(self.TEST_INPUT_IMAGE_PATH).convert('RGB')

        # ResNet-50에 사용할 전처리 및 정규화
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 이미지 전처리 및 배치 차원 추가
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)
        
        self.IN_SHAPE = img_tensor.shape
        return img_tensor


    ## ResNet50 통과 시켜서 Ouput 리턴 ##
    # Input : Image
    # Output : Feature map
    def resnet50(self):
        feature_map=None
        # ResNet-50 모델 불러오기
        resnet50_model = models.resnet50(pretrained=True)

        # 4번째 레이어까지의 모델 정의
        model_up_to_layer4 = torch.nn.Sequential(*list(resnet50_model.children())[:6])

        # 모델에 이미지 전달하여 특성 맵 얻기
        with torch.no_grad():
            model_up_to_layer4.eval()
            self.feature_map = model_up_to_layer4(self.input_image)
        
        return feature_map
    
    ## RPN 통과 시켜서 Output 리턴 ##
    # Input : Feature map
    # Output(1) : Box 좌표(batch_size, 앵커 수, 4)
    # Output(2) : 객체확률점수(batch_size, 앵커 수, 1)
    # classification을 통한 분류 후 Box 좌표만 리턴
    def rpn(self):
        bbox = None

        if not self.feature_map:
            self.resnet50()
        


        # 1. classfication

        # 2. bbox regression
        
        return bbox

    ## RoI Align Layer ##
    # Input(1) : Feature map
    # Input(2) : Box 좌표
    # Output : 고정된 크기의 Feature map(batch_size, RoI 수, 채널 수(ex: rgb == 3), pool_size, pool_size)
    def roi_align(self):
        roi_feature=None

        return roi_feature

    ## Mask Head ##
    # 고정된 크기의 Feature map을 기반으로 3개의 네트워크 구성
    # Input : RoI Align Output Feature map
    # Outputs 아래 3가지
    # 1. Class Scores(batch_size, RoI 수, 객체의 클래스 수)
    def class_scores(self):
        scores=None

        return scores

    # 2. Bounding Box Offsets(batch_size, RoI 수, 객체의 클래스 수, 4)
    def bbox_offset(self):
        o_bbox = None

        return o_bbox
    
    # 3. Mask Predictions(batch_size, RoI 수, 객체 수, 마스크 높이, 마스크 넓이)
    def mask_predict(self):
        p_mask = None

        return p_mask

if __name__ == "__main__":
    image_path = '/Users/hyeok/Desktop/Development/Python/osop/test_out/ex_mask/images/2502287818_41e4b0c4fb_z.jpg'
    test=Test_out(image_path=image_path)
    print(test.rpn())



