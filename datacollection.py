import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Open webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Settings
offset = 20
imgSize = 300
counter = 0

# Folder to store collected data
folder = "Data/Okay"
os.makedirs(folder, exist_ok=True)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure the hand is within the image bounds
        if y - offset >= 0 and y + h + offset <= img.shape[0] and x - offset >= 0 and x + w + offset <= img.shape[1]:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            aspectRatio = h / w

            # Resize maintaining aspect ratio
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hGap + hCal, :] = imgResize

            # Show images
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

            # Save image when "s" key is pressed
            key = cv2.waitKey(1)
            if key == ord("s"):
                counter += 1
                cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
                print(f"Saved: {counter}")

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
