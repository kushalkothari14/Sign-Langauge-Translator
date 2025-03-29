from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No video file provided"}), 400

    # Process the video (example using OpenCV)
    cap = cv2.VideoCapture(file.stream)
    mp_hands = mp.solutions.hands.Hands()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            return jsonify({"message": "Sign detected", "data": str(results.multi_hand_landmarks)})

    return jsonify({"message": "No sign detected"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
