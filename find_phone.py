#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Visual Object Detection using Template Matching
# Code to test on test dataset
# Please ensure that the template image created by the previous script exists in the current working directory

import cv2
import numpy as np
import os

import argparse
from pathlib import Path

# Get path to the current working directory which should contain template.jpg
cwd = os.getcwd()

# Get path to image to be tested
parser = argparse.ArgumentParser()
parser.add_argument("file_path", type=Path)

p = parser.parse_args()
#print(p.file_path, type(p.file_path), p.file_path.exists())
path = p.file_path

# Load the image
img = cv2.imread(str(path), 0)

# To hold the coordinates of the phone
label = np.zeros(2)

imgx, imgy = img.shape[::-1]
      
template = cv2.imread(str(cwd)+'/template.jpg', 0)
w, h = template.shape[::-1]

methods = 'cv2.TM_CCOEFF' 

method = eval(methods)

# Apply template Matching
res = cv2.matchTemplate(img, template, method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

mid_point = (float(top_left[0] + w/2), float(top_left[1] + h/2))

label[0] = mid_point[0]/(imgx)
label[1] = mid_point[1]/(imgy)

#print(label[0] label[1])
print(" ".join([str(label[0]), str(label[1])]))

