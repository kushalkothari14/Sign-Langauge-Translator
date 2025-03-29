import cv2
import mediapipe as mp
import pickle
import numpy as np

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M',
               12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X',
               23: 'Y', 24: 'Hello', 25: 'Thank you', 26: 'Yes', 27: 'No', 28: 'I Love you'}

while cap.isOpened():  # Ensure camera is open
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    if not ret:
        print("Error: Couldn't capture frame")
        break  # Stop if frame is not captured correctly

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect (x, y) coordinates of hand landmarks
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

        # Ensure data_aux has exactly 42 features (21 landmarks × 2)
        target_size = 42
        data_aux = np.asarray(data_aux, dtype=np.float32)

        if len(data_aux) > target_size:
            data_aux = data_aux[:target_size]  # Trim extra elements
        elif len(data_aux) < target_size:
            data_aux = np.pad(data_aux, (0, target_size - len(data_aux)), 'constant')  # Pad with zeros

        # Make prediction only if shape is correct
        if data_aux.shape[0] == 42:
            prediction = model.predict([data_aux])
            predicted_character = labels_dict[int(prediction[0])]
            print(predicted_character)
        else:
            print(f"Invalid input shape: {data_aux.shape}")


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break


cap.release()
cv2.destroyAllWindows()
