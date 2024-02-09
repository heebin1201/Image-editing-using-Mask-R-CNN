import os

# def nms(boxes, iou_threshold=0.7):
#     '''
#     boxes(list): tuple(x, y, w, h)
#     iou_threshold = 0.7 기본값
#     '''

#     if not boxes:
#         return []

#     sorted_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][2] if len(boxes[i]) > 2 else 0, reverse=True)

#     keep_indices = []
#     while sorted_indices:
#         current_index = sorted_indices[0]
#         keep_indices.append(current_index)

#         current_box = boxes[current_index]
#         remaining_indices = []
#         for i in range(1, len(sorted_indices)):
#             other_index = sorted_indices[i]
#             other_box = boxes[other_index]
#             iou = calculate_iou(current_box, other_box)
#             if iou <= iou_threshold:
#                 remaining_indices.append(other_index)

#         sorted_indices = remaining_indices

#     return keep_indices

# def calculate_iou(box1, box2):
#     '''
#     boxes(list): tuple(x, y, w, h)
#     '''

#     x1, y1, w1, h1 = box1
#     x2, y2, w2, h2 = box2

#     intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
#     area1 = w1 * h1
#     area2 = w2 * h2

#     iou = intersection_area / (area1 + area2 - intersection_area)
#     return iou


def nms(boxes, iou_threshold=0.7):
    '''
    boxes(list): tuple(x, y, w, h)
    iou_threshold = 0.7 기본값
    '''

    if not boxes:
        return []

    sorted_indices = sorted(range(len(boxes)), key=lambda i: boxes[i][2] if len(boxes[i]) > 2 else 0, reverse=True)

    keep_indices = []
    idx = []
    n = 0
    while sorted_indices:
        current_index = sorted_indices[0]
        keep_indices.append(current_index)
        current_box = boxes[current_index]
        
        remaining_indices = []

        for i in range(1, len(sorted_indices)):
            other_index = sorted_indices[i]
            other_box = boxes[other_index]
            iou = calculate_iou(current_box, other_box)
            
            if iou <= iou_threshold:
                remaining_indices.append(other_index)

        sorted_indices = remaining_indices
        selected_boxes = [boxes[i] for i in keep_indices]
        idx.append(n)
        n += 1
    return selected_boxes, idx

def calculate_iou(box1, box2):
    '''
    boxes(list): tuple(x, y, w, h)
    '''

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    area1 = w1 * h1
    area2 = w2 * h2

    iou = intersection_area / (area1 + area2 - intersection_area)
    return iou