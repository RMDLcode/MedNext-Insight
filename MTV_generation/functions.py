import SimpleITK as sitk
import numpy as np
import glob
import os
from skimage.measure import label

def largestConnectComponent(bw_img, ):
    labeled_img, num = label(bw_img, background=0, return_num=True, connectivity = 2)

    max_label = 0
    max_num = 0
    for i in range(1, num + 1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)

    return lcc, num