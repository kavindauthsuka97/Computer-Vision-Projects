import os
import cv2
import numpy as np

# read image 
image_path = 'bird.png'
img = cv2.imread(image_path)

# edge detection using canny 
img_rdge = cv2.Canny(img,100,200)


# make image edge borders thicker
img_edge_d = cv2.dilate(img_rdge, np.ones((5,5), dtype=np.int8))

img_edge_e = cv2.erode(img_edge_d, np.ones((5,5), dtype=np.int8))


# visualize the image
cv2.imshow('image',img)
cv2.imshow('img_rdge',img_rdge)
cv2.imshow('img_edge_d',img_edge_d)
cv2.imshow('img_edge_e',img_edge_e)
cv2.waitKey(0)