#!/usr/bin/env python3
"""
Data Collection Script for Sign Language Gesture Images

Uses cvzone’s HandDetector to capture and crop hand images from your webcam.
Press "s" to save the processed image and "q" to quit.
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = HandDetector(maxHands=1)

    # Settings
    offset = 20  # Padding around the hand bounding box
    imgSize = 300  # Size of the white background image
    counter = 0

    # Folder to store collected data
    folder = "Data/Okay"
    os.makedirs(folder, exist_ok=True)
    print("Data collection started. Press 's' to save an image, 'q' to quit.")

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame from webcam.")
            break

        # Find hand and get annotated image from cvzone
        hands, img = detector.findHands(img, flipType=False)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # Ensure the hand is fully within image bounds (with offset)
            if y - offset >= 0 and y + h + offset <= img.shape[0] and \
               x - offset >= 0 and x + w + offset <= img.shape[1]:

                # Create a white background image
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                # Crop the hand region from the original image with padding
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                # Resize while maintaining aspect ratio
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    newWidth = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (newWidth, imgSize))
                    wGap = math.ceil((imgSize - newWidth) / 2)
                    imgWhite[:, wGap:wGap + newWidth] = imgResize
                else:
                    k = imgSize / w
                    newHeight = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, newHeight))
                    hGap = math.ceil((imgSize - newHeight) / 2)
                    imgWhite[hGap:hGap + newHeight, :] = imgResize

                # Show the processed images
                cv2.imshow('Cropped Hand', imgCrop)
                cv2.imshow('Processed Image', imgWhite)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    counter += 1
                    filename = os.path.join(folder, f"Image_{int(time.time()*1000)}.jpg")
                    cv2.imwrite(filename, imgWhite)
                    print(f"Saved: {filename} (Total saved: {counter})")
        else:
            # If no hand is detected, show a blank white image
            cv2.imshow('Processed Image', np.ones((imgSize, imgSize, 3), np.uint8) * 255)

        cv2.imshow('Webcam Feed', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
