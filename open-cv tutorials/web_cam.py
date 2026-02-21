import cv2

# Start webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow('frame', frame)

    # Press 'q' to quit
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()quit
