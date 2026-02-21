import os
import cv2

# read image 
image_path = 'bird.png'
img = cv2.imread(image_path)

# write images already save in the memory
cv2.imwrite('bird_out.png',img)

# visualize the image
cv2.imshow('image',img)
cv2.waitKey(5000)