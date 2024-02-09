import json
import cv2
import pandas as pd
import numpy as np
# from pycocotools.coco import COCO
import streamlit as st


with open('datasets/annotations/person_keypoints_val2017.json', 'r') as file:
    data1 = json.load(file)

images = data1['images']
categories = data1['categories']
annotations1 = data1['annotations']


# print(annotations)
# print(categories)
# print(len(annotations))

ann_val = pd.DataFrame(annotations1)
print(ann_val)
# annval = ann_val.columns[3][36778]
# print(annval)
d_list=[]
print(ann_val['segmentation']==None)

for i in ann_val['segmentation']:
    if type(i) == type(list()):
        d_list.append(True)
    else:
        d_list.append(False)
dtype_list=pd.Series(d_list)

print(dtype_list)


print(ann_val[dtype_list])
print(ann_val[~ dtype_list])
