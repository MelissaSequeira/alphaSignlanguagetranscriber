import cv2 as cv
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import time
import os

# Initialize the HandDetector with maximum one hand detection
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300
counter = 0

# Root folder to save captured images
root_folder = r'C:\Users\Melissa\AppData\Local\Programs\Python\Python311\Sign_language to voice\signdata'

# Create subdirectories A-Z and 'space' if they don't exist
subdirs = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ['space']
for subdir in subdirs:
    os.makedirs(os.path.join(root_folder, subdir), exist_ok=True)


# Dictionary to keep track of saved images count for each subdirectory
image_count = {subdir: 0 for subdir in subdirs}

# Initialize the video capture
capture = cv.VideoCapture(0)

if not capture.isOpened():
    print("Couldn't detect a camera")
    exit()

while True:
    success, frame = capture.read()
    hands, frame = detector.findHands(frame)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure the crop coordinates are within the frame boundaries
        y1 = max(0, y - offset)
        y2 = min(frame.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(frame.shape[1], x + w + offset)

        if y2 > y1 and x2 > x1:
            # Create a white image
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Crop the hand region from the frame
            imgCrop = frame[y1:y2, x1:x2]
            imgCropShape = imgCrop.shape

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Display the cropped hand image and the white background image
            cv.imshow('ImageCrop', imgCrop)
            cv.imshow('ImageWhite', imgWhite)
    
    # Display the original frame with hand detection
    cv.imshow('Image', frame)

    # Check for key press
    key = cv.waitKey(1)
    if key & 0xFF == ord('q') and 'imgWhite' in locals():
        # Save images in subdirectories A-Z, each containing 5 images
        for subdir in subdirs:
            if image_count[subdir] < 5:
                counter += 1
                cv.imwrite(os.path.join(root_folder, subdir, f'Image_{time.time()}.jpg'), imgWhite)
                image_count[subdir] += 1
                print(f"Saved Image {counter} in subdirectory {subdir}")
                break  # Exit loop after saving one image

    if key & 0xFF == ord('e'):
        break

# Release the capture and destroy all OpenCV windows
capture.release()
cv.destroyAllWindows()
