import cv2                      # OpenCV: for reading images and drawing boxes/text
import easyocr                  # EasyOCR: for detecting and recognizing text in images
import matplotlib.pyplot as plt # Matplotlib: to display the image
import numpy as np              # NumPy: numerical operations (not used in this code right now)

# Path to the image you want to read (raw string avoids issues with backslashes on Windows)
image_path = r"C:\Users\HP\Desktop\Jupyter Notebooks\Projects\computer vision projects\text detection from easyocr\data\test1.png"

img = cv2.imread(image_path)    # Read the image from disk as a NumPy array (BGR format in OpenCV)

# Create an EasyOCR reader object (English language, GPU disabled)
reader = easyocr.Reader(['en'], gpu=False)

# Detect and recognize text in the image
text_ = reader.readtext(img)    # Returns a list of detections: [bbox, text, confidence_score]

threshold = 0.25                # Minimum confidence needed to draw the detection on the image

# Loop over each detected text result
for t_, t in enumerate(text_):  # t_ = index of detection, t = detection data
    print(t)                    # Print the raw detection result to the console

    bbox, text, score = t       # Unpack detection into bounding box, recognized text, confidence score

    if score > threshold:       # Only draw results with confidence above the threshold
        # Draw a green rectangle around the detected text region
        # bbox is a list of 4 points: [top-left, top-right, bottom-right, bottom-left]
        cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)  # (BGR) green box, thickness=5

        # Write the recognized text near the top-left corner of the box
        cv2.putText(
            img,                        # image to draw on
            text,                       # text to write
            bbox[0],                    # position (top-left point of bbox)
            cv2.FONT_HERSHEY_COMPLEX,   # font type
            0.65,                       # font scale (size)
            (255, 0, 0),                # (BGR) blue text color
            2                           # thickness of the text
        )

# Convert BGR (OpenCV) to RGB (Matplotlib) so colors display correctly
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.show()                    # Display the final image with boxes and recognized text