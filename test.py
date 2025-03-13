#!/usr/bin/env python3
"""
Real-Time Sign Language Gesture Recognition

Uses MediaPipe to detect the hand from your webcam, processes the image,
and predicts the gesture using a trained TensorFlow model.
Press "q" to exit.
"""

import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

def custom_depthwise_conv2d(*args, **kwargs):
    # Remove 'groups' parameter if present, as it's not needed
    kwargs.pop('groups', None)
    return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)

def main():
    # Load the trained model and labels
    model_path = "./Asl-model/asl_model.h5"
    labels_path = "./Asl-model/lasl_abels.json"
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d})
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        with open(labels_path, "r") as f:
            labels = [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error loading labels: {e}")
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    offset = 20  # Padding around detected hand
    imgSize = 300  # Size of intermediate processed image

    print("Sign language detection started. Press 'q' to quit.")

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Could not read frame from webcam.")
            break

        # Convert image to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Optionally draw landmarks on the image
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Determine the bounding box of the hand
                h, w, _ = img.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x, x_min)
                    y_min = min(y, y_min)
                    x_max = max(x, x_max)
                    y_max = max(y, y_max)

                # Apply offset to bounding box
                x_min = max(x_min - offset, 0)
                y_min = max(y_min - offset, 0)
                x_max = min(x_max + offset, w)
                y_max = min(y_max + offset, h)

                # Crop the hand region and prepare a white background image
                imgCrop = img[y_min:y_max, x_min:x_max]
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

                # Resize the cropped image while preserving aspect ratio
                handWidth = x_max - x_min
                handHeight = y_max - y_min
                aspectRatio = handHeight / handWidth

                if aspectRatio > 1:
                    k = imgSize / handHeight
                    newWidth = int(k * handWidth)
                    imgResize = cv2.resize(imgCrop, (newWidth, imgSize))
                    wGap = (imgSize - newWidth) // 2
                    imgWhite[:, wGap:wGap + newWidth] = imgResize
                else:
                    k = imgSize / handWidth
                    newHeight = int(k * handHeight)
                    imgResize = cv2.resize(imgCrop, (imgSize, newHeight))
                    hGap = (imgSize - newHeight) // 2
                    imgWhite[hGap:hGap + newHeight, :] = imgResize

                # Preprocess for model input: resize to 224x224 and normalize pixel values
                img_input = cv2.resize(imgWhite, (224, 224))
                img_input = np.expand_dims(img_input, axis=0) / 255.0

                # Get model prediction
                prediction = model.predict(img_input)
                index = np.argmax(prediction)
                label = labels[index] if index < len(labels) else "Unknown"

                # Display prediction and bounding box on the original image
                cv2.putText(img, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv2.imshow("Sign Language Detection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
