import os
import cv2

# read image 
image_path = 'bird.png'
img = cv2.imread(image_path)

# resize the image using height and width
resized_imge = cv2.resize(img, (1000,1000 ))

# print the size of the image
print(img.shape)
print(resized_imge.shape)

#visuallize the image
cv2.imshow('img',img)
cv2.imshow('resized_imge',img)

cv2.waitKey(0) # keep image visualization forever
