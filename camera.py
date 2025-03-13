#!/usr/bin/env python3
"""
Real-Time ASL Sign Language Gesture Recognition

Uses OpenCV to capture video, processes the image, and predicts the ASL sign 
using a trained TensorFlow model. Press 'q' to exit.
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import pyttsx3

def main():
    # Define model and labels path
    model_path = "./Asl-model/asl_model.h5"
    labels_path = "./Asl-model/asl_labels.json"

    # Load trained model
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load class labels from JSON file
    try:
        with open(labels_path, "r") as f:
            class_labels = json.load(f)  # Ensure it's a list
            if not isinstance(class_labels, list):
                print("Error: Labels should be a list.")
                return
    except Exception as e:
        print(f"Error loading labels: {e}")
        return

    # Initialize text-to-speech engine
    engine = pyttsx3.init()
    last_prediction = None  # Store last predicted label to avoid repeated speech

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("ASL sign language detection started. Press 'q' to quit.")

    frame_skip = 5  # Process one frame every 5 to reduce lag
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame. Retrying...")
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Skip frames to optimize performance

        # Flip frame for a mirror effect
        frame = cv2.flip(frame, 1)

        # Define region of interest (ROI) for hand detection
        roi = frame[100:400, 100:400]

        # Preprocess the ROI for prediction
        img = cv2.resize(roi, (64, 64))
        img = np.expand_dims(img, axis=0) / 255.0  # Normalize

        # Get model prediction
        prediction = model.predict(img)
        predicted_index = np.argmax(prediction)

        # **Fixed label retrieval**
        predicted_label = class_labels[predicted_index] if predicted_index < len(class_labels) else "Unknown"

        # Display the prediction
        cv2.putText(frame, f"Prediction: {predicted_label}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw rectangle around ROI
        cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)

        # Speak only if the label changes
        if predicted_label != last_prediction:
            engine.say(predicted_label)
            engine.runAndWait()
            last_prediction = predicted_label

        # Show the frame
        cv2.imshow("ASL Sign Language Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
