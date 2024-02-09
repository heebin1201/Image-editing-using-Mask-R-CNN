import cv2
import torch
import torchvision.transforms as transforms
import selectivesearch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Step 1: selective search에서 나온 박스를 하나씩 pre-trained 모델에 적용
def apply_model_to_boxes(model, image, boxes):
    results = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with torch.no_grad():
        for box in boxes:
            x, y, w, h = box
            # 추출한 박스 부분 이미지
            box_image = image[y:y+h, x:x+w]
            # 모델 입력에 맞게 변환
            input_tensor = transform(box_image)
            input_tensor = input_tensor.unsqueeze(0)  # 배치 차원 추가
            # 모델 적용
            prediction = model(input_tensor)
            results.append({
                'box': box,
                'score': prediction[0]['scores']
            })

    return results

# Step 2: 모델을 적용한 박스 중에 score값이 0.6 이상인 박스만 rp_box에 할당
def filter_boxes_by_score(boxes_results, threshold=0.6):
    rp_boxes = [result['box'] for result in boxes_results if torch.any(result['score'] >= threshold)]
    return rp_boxes

# Step 3: 할당한 rp_box를 원본 이미지에 mapping
def map_rp_boxes_to_original(image, rp_boxes):
    for rp_box in rp_boxes:
        x, y, w, h = rp_box
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Step 4: mapping한 이미지를 cv2를 이용해 시각화
def visualize_image(image):
    cv2.imshow('Visualization', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# 원본 이미지 로드
original_image = cv2.imread('D:/osop/osop/test11.jpg')

# Step 1: selective search로 후보 영역 얻기
_, regions = selectivesearch.selective_search(original_image, scale=16, min_size=800)
additional_boxes = [(region['rect'][0], region['rect'][1], region['rect'][2], region['rect'][3]) for region in regions]
# map_rp_boxes_to_original(original_image, additional_boxes)
# visualize_image(original_image)

# Step 2: 모델 적용
boxes_results = apply_model_to_boxes(model, original_image, additional_boxes)

# Step 3: Score가 0.6 이상인 박스 필터링
rp_boxes = filter_boxes_by_score(boxes_results, threshold=0.5)

# Step 4: 이미지에 박스 시각화
mapped_image = original_image.copy()
map_rp_boxes_to_original(mapped_image, rp_boxes)
visualize_image(mapped_image)
