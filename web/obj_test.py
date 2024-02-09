import numpy as np
import torch
from torchvision.transforms import transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, mask_rcnn
from PIL import Image
import random

import cv2

colors = []
for _ in range(50):
    color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
    colors.append(color)

class_names = mask_rcnn._COCO_CATEGORIES
model=maskrcnn_resnet50_fpn_v2(pretrained=True)
model.eval()
if torch.cuda.is_available():
    model.cuda()

def load_mask_rcnn(image):
    datas=[]
    
    img =np.array(image)
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    img_tensor = preprocess(img)
    if torch.cuda.is_available():
       img_tensor = img_tensor.unsqueeze(0).cuda()
    else:
        img_tensor = img_tensor.unsqueeze(0)
    
    predict = model(img_tensor)
    predict=predict[0]

    for i in range(len(predict['labels'])):
        if predict['scores'].data[i].item() > 0.8:
            data={}
            img_mask = (predict['masks'][i].squeeze(0).detach().cpu().numpy())
            
            img_mask=np.stack([img_mask]*3, axis=-1)
            data['mask']=img_mask
            data['label']=class_names[predict['labels'][i]]
            data['bbox']=predict['boxes'][i]
            datas.append(data)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img, datas

def bbox_draw(image, datas):
    box_img=image.copy()
    for i in range(len(datas)): 
        x,y,w,h=datas[i]['bbox'].detach().numpy().astype('int')

        cv2.rectangle(box_img,(x,y),(w,h),colors[i],5)
        cv2.rectangle(box_img,(x-2, y),(x+275,y-90),colors[i],-1)
        cv2.putText(box_img,str(datas[i]['label']+str(i+1)),(x+10, y-25),cv2.FONT_HERSHEY_SIMPLEX,2.0,(255,255,255),5)
    
    return box_img

def mask_draw(image, datas):
    mask_img=image.copy()
    for i in range(len(datas)):
        mask = datas[i]['mask']
        if isinstance(mask, np.ndarray):
                _,mask_color=cv2.threshold(mask,0.5,255,cv2.THRESH_BINARY)
                
                # mask_color=cv2.imdecode()
                for j in range(3):
                    mask_color[mask_color[..., j] == 255] = colors[i]

                alpha = ((mask_color >0).max(axis=2)*128).astype(np.uint8)
                rgba_mask = np.concatenate([mask_color, alpha[:,:, np.newaxis]], axis=2)
                image_rgba = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGBA)
                image_rgba = cv2.addWeighted(image_rgba, 1, rgba_mask, 0.7, 0, dtype = cv2.CV_8U)

                mask_img = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGR)

    return mask_img

image = Image.open('web/assets/test1.jpeg').convert('RGB')
img, datas = load_mask_rcnn(image)

# cv2.imshow('img',img)
# print(datas)
box_img=bbox_draw(img, datas)
cv2.imshow('bbox', box_img)
cv2.waitKey(0)
cv2.imwrite('web/test/box_img.jpg',box_img)

mask_img=mask_draw(img, datas)
cv2.imshow('mask', mask_img)
cv2.waitKey(0)
cv2.imwrite('web/test/mask_img.jpg',mask_img)
cv2.destroyAllWindows()