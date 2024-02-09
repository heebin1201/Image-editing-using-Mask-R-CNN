import cv2
import selectivesearch
import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - box1, box2: List or tuple representing (x1, y1, x2, y2) coordinates of the boxes.

    Returns:
    - IoU: Intersection over Union value.
    """

    # Extract coordinates
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    # Calculate the intersection area
    x_overlap = max(0, min(x2_box1, x2_box2) - max(x1_box1, x1_box2))
    y_overlap = max(0, min(y2_box1, y2_box2) - max(y1_box1, y1_box2))
    intersection_area = x_overlap * y_overlap

    # Calculate the area of each bounding box
    area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
    area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)

    # Calculate the Union area
    union_area = area_box1 + area_box2 - intersection_area

    # Avoid division by zero
    if union_area == 0:
        return 0.0

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def nms(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression to eliminate overlapping bounding boxes.

    Parameters:
    - boxes: List of bounding boxes, each represented as (x1, y1, x2, y2).
    - scores: List of confidence scores associated with each bounding box.
    - iou_threshold: IoU threshold for considering boxes as overlapping.

    Returns:
    - selected_boxes: List of selected bounding boxes after NMS.
    - selected_scores: List of corresponding confidence scores.
    """

    # Sort boxes based on their scores in descending order
    sorted_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

    selected_boxes = []
    selected_scores = []

    while len(sorted_indices) > 0:
        # Pick the box with the highest score
        best_index = sorted_indices[0]
        selected_boxes.append(boxes[best_index])
        selected_scores.append(scores[best_index])

        # Remove the current box from the list
        del sorted_indices[0]

        # Calculate IoU with the remaining boxes
        iou_values = [calculate_iou(boxes[best_index], boxes[i]) for i in sorted_indices]

        # Remove boxes with IoU greater than the threshold
        sorted_indices = [i for i in range(len(sorted_indices)) if iou_values[i] <= iou_threshold]

    return selected_boxes, selected_scores


img = cv2.imread('D:/osop/osop/test12.jpg')
_, regions = selectivesearch.selective_search(img,scale=100 , min_size=2000)

boxes = [i['rect'] for i in regions]
scores = [i['size'] for i in regions]
print(boxes)
selected_boxes, selected_scores = nms(boxes, scores, 0.5)

for box in selected_boxes:
    x, y, w, h = box
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('t', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
