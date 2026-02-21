import os
import cv2

img = 'bird.png'
img = cv2.imread(img)

# convert image from BGR color space to RGB color space
# BGR -> Blue Green Red
# RGB -> Red Green Blue
# everything same just change how we organize the color changes
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# convert BGR color space to gray scale. we convert three channel information into only one channel
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# convert to HSV color space
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('img',img)
cv2.imshow('img_rgb',img_rgb)
cv2.imshow('img_gray',img_gray)
cv2.imshow('img_hsv',img_hsv)
cv2.waitKey(0)