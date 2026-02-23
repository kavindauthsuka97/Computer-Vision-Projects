import os
import cv2

# read image 
image_path = 'bird.png'
img = cv2.imread(image_path)

# draw a line
# first () is starting point of the line
# second () is ending point of the line
# 3 is the thickness of the lines. if we give -1 for it , it will be colored the circled
# (0,255,0) this is the color channel blue, green, red
cv2.line(img, (100,150) , (300,450), (0,255,0),3)

# draw a rectangle
cv2.rectangle(img, (200,350), (450,600), (0,0,255),-1 )

# draw a circle
# first () is the center coordinates of the circle
cv2.circle(img, (500,550), 15 , (0,255,0))

# add text on top of an image
cv2.putText(img, 'hey you', (800,450),cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,0),2)

# visualize the image
cv2.imshow('image',img)
cv2.waitKey(5000)