import datetime
import math
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import torchvision




class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__
  
    
# mask부분의 conv2d는 input으로 (7,7) 사이즈의 피쳐맵을 받음.
# 피쳐맵은 roialign을 통과해 나온 output값으로,
# [batch_size,num_rois,channels,pool_height, pool_width]의 값을 가지고 있다. 
# 단, test할때 단일 이미지를 가지고 하는경우 batch_size를 생략해야하고, 생성자부분도
# 체크해서 생략해 주어야 함.
# 도식상, roi_align에서 나온 값을 256채널로 받고, 이후 80(num_classes)로 내보낸 것으로
# 파악. num_classes = 80인 이유는 기존의 모델이 coco데이터 셋을 통한 학습 모델이며,
# coco데이터셋을 기준으로 num_classes의 수가 80으로 설정되어있는것으로 파악
# 추가적으로 deconv층은 효율에 쓸지 안쓸지 여부를 결정하면 될것으로 파악됨.
# 모델의 도식을 고려하여, 추가적으로 deconv층을 주석처리 하였음.

# class Mask(nn.Module):
#     def __init__(self, batch_size,num_rois,in_channels,pool_height,pool_weight,num_classes):
#         super(Mask, self).__init__()
#         self.batch_size = batch_size
#         self.num_rois = num_rois
#         self.in_channels = in_channels
#         self.num_classes = num_classes
#         self.pool_height = pool_height
#         self.pool_weight = pool_weight
#         self.padding = SamePad2d(kernel_size=3,stride=1)
#         self.conv1 = nn.Conv2d(self.in_channels, 256, kernel_size=3, stride=1)
#         self.bn1 = nn.BatchNorm2d(256, eps=0.001)
#         # self.deconv = nn.ConvTranspose2d(256, 80, kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(256,self.num_classes, kernel_size=3, stride=1)
#         self.sigmoid = nn.Sigmoid()
#         self.relu = nn.ReLU(inplace=True)
        
#     #mask의 input은 roi aling의 output 중에, batch_size,rois,num_classes,pool_height,pool_weight만
#     #input으로 받으면 된다.
#     #forward의 첫번째 x는 roi_align의 아웃풋을 받는것으로, 설정해두었다.
#     def forward(self, x):
#         x = self.conv1(self.padding(x))
#         x = self.bn1(x)
#         x = self.relu(x)
#         # x = self.deconv(x)
#         x = self.conv2(self.padding(x))
#         x = self.sigmoid(x)
#         p_mask = x
#         return p_mask
    
class SamePad2d(nn.Module):
    """Mimics tensorflow's 'SAME' padding.
    """

    def __init__(self, kernel_size, stride):
        super(SamePad2d, self).__init__()
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        self.stride = torch.nn.modules.utils._pair(stride)

    def forward(self, input):
        in_width = input.size()[2]
        in_height = input.size()[3]
        out_width = math.ceil(float(in_width) / float(self.stride[0]))
        out_height = math.ceil(float(in_height) / float(self.stride[1]))
        pad_along_width = ((out_width - 1) * self.stride[0] +
                           self.kernel_size[0] - in_width)
        pad_along_height = ((out_height - 1) * self.stride[1] +
                            self.kernel_size[1] - in_height)
        pad_left = math.floor(pad_along_width / 2)
        pad_top = math.floor(pad_along_height / 2)
        pad_right = pad_along_width - pad_left
        pad_bottom = pad_along_height - pad_top
        return F.pad(input, (pad_left, pad_right, pad_top, pad_bottom), 'constant', 0)

    def __repr__(self):
        return self.__class__.__name__

class Mask(nn.Module):
  def __init__(self, num_classes):
      super(Mask, self).__init__()
      self.num_classes = num_classes
      self.padding = SamePad2d(kernel_size=3,stride=1)
      self.conv1 = nn.Conv2d(2048, 256, kernel_size=3, stride=1)
      self.bn1 = nn.BatchNorm2d(256, eps=0.001)
      self.deconv = nn.ConvTranspose2d(256, 80, kernel_size=2, stride=2)
      self.conv2 = nn.Conv2d(80,self.num_classes, kernel_size=3, stride=1)
      self.sigmoid = nn.Sigmoid()
      self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.conv1(self.padding(x))
    x = self.bn1(x)
    x = self.relu(x)
    x = self.deconv(x)
    x = self.conv2(self.padding(x))
    x = self.sigmoid(x)
    p_mask = x
    return p_mask
