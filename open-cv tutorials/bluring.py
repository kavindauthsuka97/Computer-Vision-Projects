import os
import cv2

# read image 
image_path = 'bird.png'
img = cv2.imread(image_path)

# classical blur
k_size = 7 # number of neighborhood pixels that needs to blur. when we increase the size of this number power of blur will be increased.
img_blur = cv2.blur(img, (k_size,k_size))

# gaussian blur
# all the parameters are same as classical blur
# 5 is the sigmax value. bigger sigmax gives stronger blur
img_gblur = cv2.GaussianBlur(img, (k_size,k_size),5)


# median blur



# visualize the image
cv2.imshow('image',img)
cv2.imshow('img_blur',img_blur)
cv2.imshow('img_gblur',img_gblur)
cv2.waitKey(0)