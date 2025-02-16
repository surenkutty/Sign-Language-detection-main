import cv2
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf
import numpy as np
import math

# Load the trained model
model = tf.keras.models.load_model("./Model/keras_model.h5")

# Read labels
with open("./Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Open webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Settings
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure the hand is within bounds
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

            # Preprocess the image for the model
            imgWhite = cv2.resize(imgWhite, (224, 224))  # Ensure it matches model input size
            imgWhite = np.expand_dims(imgWhite, axis=0) / 255.0  # Normalize

            # Make prediction
            prediction = model.predict(imgWhite)
            index = np.argmax(prediction)

            # Display results
            label = labels[index]
            print(f"Detected: {label}")

            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, label, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
