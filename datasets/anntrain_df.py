import json
import cv2
import pandas as pd
import numpy as np
from pycocotools.coco import COCO

with open('instances_train2017.json', 'r') as file:
    data2 = json.load(file)

annotations2 = data2['annotations']

ann_train = pd.DataFrame(annotations2)
print(ann_train)