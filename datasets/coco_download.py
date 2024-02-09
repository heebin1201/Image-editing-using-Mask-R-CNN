from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

dataDir = 'datasets'
dataType = 'val2017'

# initialize COCO api for person keypoints annotations
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
coco = COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories : \n{}\n'.format(' '.join(nms)))

nms = [cat['supercategory'] for cat in cats]
print('COCO supercategories : \n{}\n'.format(' '.join(nms)))

