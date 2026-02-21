'''
thresholding means converting the imafe into a binary variable
'''

import os
import cv2

# read image 
image_path = 'bird.png'
img = cv2.imread(image_path)

# 1- convert color space to gray color space
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2 - simple thresholding
# 80 means all the values which are less than 80 goes to zero and all the values which are greater than goes to 255
ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)

# adapitve threshold. this is more powerful than simple threshold
ada_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21 , 30)


# visualize the image
cv2.imshow('image',img)
cv2.imshow('thresh',thresh)
cv2.imshow('ada_thresh',ada_thresh)
cv2.waitKey(5000)