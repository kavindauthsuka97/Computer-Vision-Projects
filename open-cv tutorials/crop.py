import os
import cv2

# read image 
image_path = 'bird.png'
img = cv2.imread(image_path)

print(img.shape)

# crop the image according to the height and weight. image cropping means slice the numoy arrays
cropped_img = img[320:400,300:360]




#visuallize the image
cv2.imshow('img',img)
cv2.imshow('cropped_img',cropped_img)
cv2.waitKey(0) # keep image visualization forever