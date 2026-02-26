'''
process
    1 - read images
    2 - detect faces
    3 - blur faces
    4 - save image
'''

import os           # for file and directory operations
import argparse     # for parsing command-line arguments

import cv2          # OpenCV for image and video processing
import mediapipe as mp  # MediaPipe for face detection


def process_img(img, face_detection):

    H, W, _ = img.shape    # get image height and width

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR image to RGB (MediaPipe requires RGB)
    out = face_detection.process(img_rgb)            # run face detection on the RGB image

    if out.detections is not None:                   # check if any faces were detected
        for detection in out.detections:             # loop through each detected face
            location_data = detection.location_data  # get location data of the face
            bbox = location_data.relative_bounding_box  # get bounding box (values are relative 0-1)

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height  # unpack bounding box values

            x1 = int(x1 * W)   # convert relative x position to pixel value
            y1 = int(y1 * H)   # convert relative y position to pixel value
            w = int(w * W)      # convert relative width to pixel value
            h = int(h * H)      # convert relative height to pixel value

            # blur the detected face region using a 30x30 kernel
            img[y1:y1 + h, x1:x1 + w, :] = cv2.blur(img[y1:y1 + h, x1:x1 + w, :], (30, 30))

    return img  # return the processed image with blurred faces


args = argparse.ArgumentParser()    # create argument parser object

args.add_argument("--mode", default='webcam')       # add mode argument (image, video, or webcam)
args.add_argument("--filePath", default=None)       # add file path argument for image/video input

args = args.parse_args()    # parse the command-line arguments


output_dir = './output'                 # define output directory path
if not os.path.exists(output_dir):      # check if output directory doesn't exist
    os.makedirs(output_dir)             # create the output directory

# initialize MediaPipe face detection solution
mp_face_detection = mp.solutions.face_detection

# create face detection context (model_selection=0 for short-range detection)
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:

    if args.mode in ["image"]:
        img = cv2.imread(args.filePath)             # read the image from the given file path

        img = process_img(img, face_detection)      # detect and blur faces in the image

        cv2.imwrite(os.path.join(output_dir, 'output.png'), img)    # save the processed image to output directory

    elif args.mode in ['video']:

        cap = cv2.VideoCapture(args.filePath)       # open the video file
        ret, frame = cap.read()                     # read the first frame

        # create video writer to save the output video as MP4 at 25 FPS
        output_video = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'),
                                       cv2.VideoWriter_fourcc(*'MP4V'),     # set codec to MP4V
                                       25,                                   # set frame rate to 25 FPS
                                       (frame.shape[1], frame.shape[0]))     # set frame size

        while ret:                                          # loop until no more frames
            frame = process_img(frame, face_detection)      # detect and blur faces in the frame
            output_video.write(frame)                       # write the processed frame to output video
            ret, frame = cap.read()                         # read the next frame

        cap.release()           # release the video capture object
        output_video.release()  # release the video writer object

    elif args.mode in ['webcam']:
        cap = cv2.VideoCapture(2)   # open the webcam (device index 2)

        ret, frame = cap.read()     # read the first frame from webcam
        while ret:                                          # loop until webcam stops
            frame = process_img(frame, face_detection)      # detect and blur faces in the frame

            cv2.imshow('frame', frame)  # display the processed frame in a window
            cv2.waitKey(25)             # wait 25ms between frames (~40 FPS)

            ret, frame = cap.read()     # read the next frame from webcam

        cap.release()   # release the webcam