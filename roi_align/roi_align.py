import os

import torch
import torch.nn.functional as F
from functorch.dim import dims
import math

#원본: https://gist.github.com/ezyang/813e86ff5b46ae9e41fc1920790e51fc
#쌍선형 보간: 선형 보간을 2차원으로 확장하여 임의의 점의 함수값을 구하는 방법
"""
input:[batch_size, channels, width, height] / dtype:Tensor.float
x,y: 쌍선형 보간을 수행할 좌표 / dtype: tensor / shape=(2, 7, 28) (*(n, pw, ix))
width, height: 피쳐맵의 너비, 높이 / dtype: int
xmask, ymask: x,y 좌표에 대한 이진 마스크(특정 위치에서 보간을 정확하게 수행할 수 있도록 해주는 역할) / dtype: Tensor(bools) / shape:(28, 2)

output:[n, ph, iy, pw, ix, c] (*shape=(2, 7, 28, 7, 28, 256)) / dtype:Tensor.float
(*n: batch_size, ph,pw:자를 높이,너비, ix,iy: 주어진 좌표 주변 4개의 점 좌표)
"""

def bilinear_interpolate(input, x, y, width, height, xmask,  ymask):
    y = y.clamp(min=0)
    x = x.clamp(min=0)
    y_low = y.int()
    x_low = x.int()
    y_high = torch.where(y_low >= height - 1, height - 1, y_low + 1)
    y_low = torch.where(y_low >= height - 1, height - 1, y_low)
    y = torch.where(y_low >= height - 1, y.to(input.dtype), y)

    x_high = torch.where(x_low >= width - 1, width - 1, x_low + 1)
    x_low = torch.where(x_low >= width - 1, width - 1, x_low)
    x = torch.where(x_low >= width - 1, x.to(input.dtype), x)

    ly = y - y_low
    lx = x - x_low
    hy = 1. - ly
    hx = 1. - lx

    #보간에 사용되는 픽셀이 유효한지 판단 하는 함수
    def masked_index(y, x):
        y = torch.where(ymask, y, 0)
        x = torch.where(xmask, x, 0)
        return input[y, x]

    v1 = masked_index(y_low, x_low)
    v2 = masked_index(y_low, x_high)
    v3 = masked_index(y_high, x_low)
    v4 = masked_index(y_high, x_high)
    w1 = hy * hx
    w2 = hy * lx
    w3 = ly * hx
    w4 = ly * lx

    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4

    return val


"""
<roi align함수 매개변수>
input:[batch_size, channels,width, height] / dtype:Tensor.float
rois: [batch_index, x1,y1,x2,y2] (참고:왼쪽위 x,왼쪽위 y, 오른쪽아래 x,오른쪽아래 y) / dtype:Tensor.float
spatial_scale:축소 비율 / dtype: float
polled_width, hight: 출력 피쳐맵의 너비, 높이 / dtype: int / 7,7로 기본 설정해둠
sampling_ratio: RoI 내에서 각 bin에서 얼마나 많은 샘플을 추출할지를 나타내는 비율 / -1로 기본 설정해둠
aligned: false이면 legacy implementation 사용/ False로 기본 설정해둠

<출력>
#output: The pooled RoIs:잘린 피쳐맵[n,c,pw,ph] => dtype: tensor.float

"""
# def roi_align(input, rois, spatial_scale, pooled_width=7, pooled_height=7, sampling_ratio=-1, aligned=False):
#     _, _, height, width = input.size()

#     n, c, ph, pw = dims(4)
#     #print(n,c,ph,pw)
#     ph.size = pooled_height
#     pw.size = pooled_width
#     offset_rois = rois[n]
#     roi_batch_ind = offset_rois[0].int()
#     offset = 0.5 if aligned else 0.0
#     roi_start_w = offset_rois[1] * spatial_scale - offset
#     roi_start_h = offset_rois[2] * spatial_scale - offset
#     roi_end_w = offset_rois[3] * spatial_scale - offset
#     roi_end_h = offset_rois[4] * spatial_scale - offset

#     roi_width = roi_end_w - roi_start_w
#     roi_height = roi_end_h - roi_start_h
#     if not aligned:
#         roi_width = torch.clamp(roi_width, min=1.0)
#         roi_height = torch.clamp(roi_height, min=1.0)

#     bin_size_h = roi_height / pooled_height
#     bin_size_w = roi_width / pooled_width

#     offset_input = input[roi_batch_ind][c]

#     roi_bin_grid_h = sampling_ratio if sampling_ratio > 0 else torch.ceil(roi_height / pooled_height)
#     roi_bin_grid_w = sampling_ratio if sampling_ratio > 0 else torch.ceil(roi_width / pooled_width)

#     count = torch.clamp(roi_bin_grid_h * roi_bin_grid_w, min=1)

#     iy, ix = dims(2)

#     iy.size = height  # < roi_bin_grid_h
#     ix.size = width  # < roi_bin_grid_w

#     #roi 공간에 따른 픽셀 위치 설정
#     y = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
#     x = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w
#     ymask = iy < roi_bin_grid_h
#     xmask = ix < roi_bin_grid_w

#     #쌍선형 보간 수행
#     val = bilinear_interpolate(offset_input, x, y, width, height, xmask, ymask)

#     #유효한 위치에 대한 마스킹 및 피쳐 맵 풀링
#     val = torch.where(ymask, val, 0)
#     val = torch.where(xmask, val, 0)
#     output = val.sum((iy, ix))
#     output /= count
#     return output.order(n, c, pw, ph)

#테스트 코드
# This could be an output from a convolutional layer of a CNN
# features = torch.ones(1, 256, 28,28)

# rois = torch.tensor([
#     [0, 60, 60, 100, 100],    #[이미지의 인덱스, 왼쪽위 x,왼쪽위 y, 오른쪽아래 x,오른쪽아래 y] *바운딩 박스의 x,y죄표
#     [0, 120, 120, 160, 160]
# ], dtype=torch.float)
# print(rois.shape)
# spatial_scale = 1.0 / 8.0

# Call the roi_align function
# pooled_features = roi_align(features, rois, spatial_scale)
# output_size = 7
# print(f"제작:{pooled_features.shape}")

#기작성된 함수 불러섭 비교
#from torchvision.ops import roi_align as roi_align_torchvision
#print(f"파이토치 함수:{roi_align_torchvision(features, rois, output_size, spatial_scale).sum()}")


def roi_align(feature, rois, spatial_scale, pooled_width=16, pooled_height=16, sampling_ratio=-1, aligned=False):
    _, _, height, width = feature.size()

    n, c, ph, pw = dims(4)
    ph.size = pooled_height
    pw.size = pooled_width
    offset_rois = rois[n]
    roi_batch_ind = offset_rois[0].int()
    offset = 0.5 if aligned else 0.0
    roi_start_w = offset_rois[1] * spatial_scale - offset
    roi_start_h = offset_rois[2] * spatial_scale - offset
    roi_end_w = offset_rois[3] * spatial_scale - offset
    roi_end_h = offset_rois[4] * spatial_scale - offset

    roi_width = roi_end_w - roi_start_w
    roi_height = roi_end_h - roi_start_h
    if not aligned:
        roi_width = torch.clamp(roi_width, min=1.0)
        roi_height = torch.clamp(roi_height, min=1.0)

    bin_size_h = roi_height / pooled_height
    bin_size_w = roi_width / pooled_width

    offset_input = feature[roi_batch_ind][c]

    roi_bin_grid_h = sampling_ratio if sampling_ratio > 0 else torch.ceil(roi_height / pooled_height)
    roi_bin_grid_w = sampling_ratio if sampling_ratio > 0 else torch.ceil(roi_width / pooled_width)

    count = torch.clamp(roi_bin_grid_h * roi_bin_grid_w, min=1)

    iy, ix = dims(2)

    iy.size = height  # < roi_bin_grid_h
    ix.size = width  # < roi_bin_grid_w

    #roi 공간에 따른 픽셀 위치 설정
    y = roi_start_h + ph * bin_size_h + (iy + 0.5) * bin_size_h / roi_bin_grid_h
    x = roi_start_w + pw * bin_size_w + (ix + 0.5) * bin_size_w / roi_bin_grid_w
    ymask = iy < roi_bin_grid_h
    xmask = ix < roi_bin_grid_w

    #쌍선형 보간 수행
    val = bilinear_interpolate(offset_input, x, y, width, height, xmask, ymask)

    #유효한 위치에 대한 마스킹 및 피쳐 맵 풀링
    val = torch.where(ymask, val, 0)
    val = torch.where(xmask, val, 0)
    output = val.sum((iy, ix))
    output /= count
    return output.order(n, c, pw, ph)