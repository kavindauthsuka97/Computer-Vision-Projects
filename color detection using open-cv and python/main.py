import cv2
from PIL import Image

from util import get_limits


yellow = [0, 255, 255]  # yellow in BGR colorspace
cap = cv2.VideoCapture(0) # webcam that we need to open
# read the video frame by frame
while True:
    ret, frame = cap.read()
    # convert color space from BGR to HSV
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # mention the color
    lowerLimit, upperLimit = get_limits(color=yellow)
    # get a mask from all the pixels that we want to predict
    mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    
    mask_ = Image.fromarray(mask)

    # draw bounding box around the yellow color item
    bbox = mask_.getbbox()

    # draw a rectangle using bounding boxes
    if bbox is not None:
        x1, y1, x2, y2 = bbox

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()