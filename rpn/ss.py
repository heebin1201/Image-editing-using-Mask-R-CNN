import selectivesearch
import os
import numpy as np

# def ss(img):
#     # Selective Search를 통해 region proposal 수행
#     _, regions = selectivesearch.selective_search(img, scale=100, min_size=2000)
#     rects = [cand['rect'] for cand in regions]
#     return rects

def selective_search(img):
  img = np.array(img)
  _, regions = selectivesearch.selective_search(img, scale=100, min_size=2000)
  rects = [cand['rect'] for cand in regions]
  return rects