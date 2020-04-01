#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Visual Object Detection using Template Matching 
# Code to train on training dataset
## There is no machine learning involved in this method and it works really well. I've attached more details and suggestions
## in a pdf file named "Notes"
### !apt-get -qq install -y libsm6 libxext6 && pip install -q -U opencv-python (To install OpenCV on Google Colab)

import cv2
import numpy as np
import argparse
from pathlib import Path

# Accept path from command line
parser = argparse.ArgumentParser()
parser.add_argument("file_path", type=Path)

p = parser.parse_args()

path = p.file_path

# Store the labels in a dictionary
newDict = {}
with open(str(path)+"/labels.txt", 'r') as f:
        lines = f.readlines()
        k = 0
        for i in lines:
          words = i.split()
          temp = words[0]
          img_name = int(temp[0:len(temp)-4])
          coordx = float(words[1])
          coordy = float(words[2])
          newDict[img_name] = (coordx, coordy)
            
# Create Template Image
img = cv2.imread(str(path)+"/"+"0.jpg", 0)
imgx, imgy = img.shape[::-1]
mx, my = newDict[0]
mx = int(mx*imgx)
my = int(my*imgy)
temp = img[my-40:my+40, mx-37:mx+37]
cv2.imwrite(str(path)+"/template.jpg", temp)

# Create a variable to store labels after template matching
label = np.zeros((135, 2))

# Iterate over each image in the folder and generate the label
for i in range(0, 135):

    # These images were missing in the given dataset
    if i in [2, 19, 21, 28, 56, 65]:
      label[i][0] = 0
      label[i][1] = 0
      
    else:
      img = cv2.imread(str(path)+"/"+str(i)+".jpg", 0)
      imgx, imgy = img.shape[::-1]
      
      template = cv2.imread(str(path)+'/template.jpg', 0)
      w, h = template.shape[::-1]

      methods = 'cv2.TM_CCOEFF' 

      method = eval(methods)

    # Apply template Matching
      res = cv2.matchTemplate(img, template, method)
    
      min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
      top_left = max_loc

      bottom_right = (top_left[0] + w, top_left[1] + h)

      mid_point = (float(top_left[0] + w/2), float(top_left[1] + h/2))

      label[i][0] = mid_point[0]/(imgx)
      label[i][1] = mid_point[1]/(imgy)

      cv2.rectangle(img, top_left, bottom_right, 255, 2)
      #cv2.imwrite(str(path)+"/output_images/"+str(i)+"_out.jpg", img)
      cv2.imwrite(str(path)+"/"+str(i)+"_out.jpg", img)
  
# Compute the total valid matches
count = 0
for i in range(0, 135):
  if i in [2, 19, 21, 28, 56, 65]:
    newDict[i] = (0, 0)
  accx, accy = label[i] - newDict[i]
  if(accx < 0.05 and accy < 0.05):
    count += 1
   
# Accuracy of prediction in training
print(100*count/134)

