import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import matplotlib.pyplot as plt


class COCO_dataformat(Dataset):
    def __init__(self, data_dir, mode = 'train', transform = None, dataset_path='datasets/images/train2017/'):
        self.dataset_path = dataset_path
        super().__init__()
        self.mode = mode
        # transform : albumentations 라이브러리 사용
        self.transform = transform
        self.coco = COCO(data_dir)

        # category id 반환
        self.cat_ids = self.coco.getCatIds()
        # category id를 입력으로 category name, super category 정보 담긴 dict 반환
        self.cats = self.coco.loadCats(self.cat_ids)
        # class name 저장 
        self.classNameList = ['Backgroud']
        for i in range(len(self.cat_ids)):
          self.classNameList.append(self.cats[i]['name'])

    def __getitem__(self, index: int):
        # img id 또는 category id 를 받아서 img id 반환
        image_id = self.coco.getImgIds(imgIds=index)
        print(image_id)
        # img id를 받아서 image info 반환
        image_infos = self.coco.loadImgs(image_id)[0]
        print(os.path.join(self.dataset_path, image_infos['file_name']))
        
        # cv2 를 활용하여 image 불러오기(BGR -> RGB 변환 -> numpy array 변환 -> normalize(0~1))
        images = cv2.imread(os.path.join(self.dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB).astype(np.float32)
        # albumentations 라이브러리로 toTensor 사용시 normalize 안해줘서 미리 해줘야~
        images /= 255.0
        
        image_infos = self.coco.loadImgs(image_id)[0]
        if (self.mode in ('train', 'val')):
            #img id, category id를 받아서 해당하는 annotation id 반환
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            # annotation id를 받아서 annotation 정보 반환
            anns = self.coco.loadAnns(ann_ids)
            print(anns)

            # 저장된 annotation 정보로 label mask 생성, Background = 0, 각 pixel 값에는 "category id" 할당
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            anns = sorted(anns, key=lambda idx : len(idx['segmentation'][0]), reverse=False)
            # 이미지 하나에 존재하는 annotation 반복
            image_infos = self.coco.loadImgs(image_id)[0]
            for i in range(len(anns)):
                # 해당 클래스 이름의 인덱스
                #className = classNameList[anns[i]['category_id']] # 클래스 이름
                pixel_value = anns[i]['category_id']
                # coco.annToMask(anns) : anns 정보로 mask를 생성 / 객체가 있는 곳마다 객체의 label에 해당하는 mask 생성
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)
                        
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos
        
        if self.mode == 'test':
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos
    
    def __len__(self) -> int:
        # 전체 dataset의 size 반환 
        return len(self.coco.getImgIds())

import albumentations as A
from albumentations.pytorch import ToTensorV2

def collate_fn(batch):
    return tuple(zip(*batch))

def dataset_set(batch_size = 8):
    train_path = 'datasets/annotations/instances_train2017.json'
    val_path = 'datasets/annotations/instances_val2017.json'
    test_path = 'datasets/annotations/image_info_test2017.json'
    train_transform = A.Compose([
                                ToTensorV2()
                                ])

    val_transform = A.Compose([
                            ToTensorV2()
                            ])

    test_transform = A.Compose([
                            ToTensorV2()
                            ])

    train_dataset = COCO_dataformat(data_dir=train_path, mode='train', transform=train_transform)
    val_dataset = COCO_dataformat(data_dir=val_path, mode='val', transform=val_transform)
    test_dataset = COCO_dataformat(data_dir=test_path, mode='test', transform=test_transform)

    
    return train_dataset, val_dataset, test_dataset

if __name__ == "__main__":
    batch_size = 8
    train_dataset, val_dataset, test_dataset = dataset_set()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            num_workers=4,
                                            collate_fn=collate_fn)

    numOfSample = 3
    fig, axes = plt.subplots(nrows=numOfSample, ncols=2, figsize=(15, 15))
    tmp_imgs = []
    tmp_masks = []
    coco = COCO('datasets/annotations/instances_train2017.json')

    for imgs, image_info in train_loader:

        print(imgs)
        break

    # for imgs, masks, image_infos in train_loader:
    #     for idx in range(numOfSample):
    #         tmp_info = []
    #         for anns in coco.loadAnns(coco.getAnnIds(image_infos[idx]['id'])) :
    #             if [anns['category_id'],train_dataset.classNameList[anns['category_id']]] not in tmp_info:
    #                 tmp_info.append([anns['category_id'],train_dataset.classNameList[anns['category_id']]])
    #         print(idx,'번째 이미지에 포함된 labels :',tmp_info)
    #         axes[idx][0].imshow(imgs[idx].permute([1,2,0]))
    #         axes[idx][0].grid(False)
    #         axes[idx][0].set_title("input image : {}".format(image_infos[idx]['file_name']), fontsize = 15)

    #         axes[idx][1].imshow(masks[idx])
    #         axes[idx][1].grid(False)
    #         axes[idx][1].set_title("input image : {}".format(image_infos[idx]['file_name']), fontsize = 15)
    #         break

    plt.show()