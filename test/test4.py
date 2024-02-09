import cv2
import torch
import torchvision.transforms as transforms
import selectivesearch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import numpy as np

# 모델 정의
weight = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weight=weight,pretrained=True)
model.eval()
model.cuda()

# 이미지 전처리 함수
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# 이미지 로드
og_img = cv2.imread('D:/osop/osop/test11.jpg')

# Selective Search로 후보 영역 얻기
_, regions = selectivesearch.selective_search(og_img, scale=100, min_size=2000)

# 모델을 적용한 박스 중에 score값이 0.6 이상인 박스만 추출
boxes = []
scores = []
for rect in regions:
    x, y, w, h = rect['rect']
    box_image = og_img[y:y+h, x:x+w]
    # cv2.imshow('ss',box_image)
    # cv2.waitKey(0)
    input_tensor = preprocess_image(box_image)
    
    with torch.no_grad():
        prediction = model(input_tensor.cuda())
    # print(prediction)
    # print(np.argmax(prediction, axis=1))

    if torch.any(prediction[0]['scores'] >= 0.5):
        boxes.append(prediction[0]['boxes'].cpu().numpy())
        scores.append(prediction[0]['scores'].cpu().numpy())
# print(scores)

indices=[]
for i in range(len(boxes)):
    # indices.append(cv2.dnn.NMSBoxes(boxes[i],scores[i],score_threshold=0.5,nms_threshold=0.2))
    for j in range(len(boxes[i])):
        if scores[i][j] >= 0.5:
            x, y, w, h = boxes[i][j]
            cv2.rectangle(og_img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)


# print(prediction)
# for pred in prediction:
#     boxes=pred['boxes'].cpu().numpy()
#     scores=pred['scores'].cpu().numpy()
#     # print(boxes)
#     # print(scores)
#     for i in range(len(boxes)):
#         if scores[i] > 0.6:
#             x, y, w, h = boxes[i]
#             cv2.rectangle(og_img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
# cv2.imshow('pre_box',og_img)
# cv2.waitKey(0)
    # print(indices)

# print(indices)

# NMS 적용
# for indice in indices:
#     # NMS를 통과한 박스만 남기기
#     filtered_boxes = [boxes[i] for i in indice]

#     # 할당한 rp_box를 원본 이미지에 mapping

#     # mapped_image = og_img.copy()
#     for x, y, w, h in filtered_boxes:
#         print(x,y,w,h)
#         cv2.rectangle(og_img, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)

# mapping한 이미지를 cv2를 이용해 시각화
cv2.imshow('Mapped Image with NMS', og_img)
cv2.waitKey(0)
cv2.destroyAllWindows()