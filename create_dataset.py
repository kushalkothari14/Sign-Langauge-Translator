import os
import pickle
import mediapipe as mp
import cv2


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


DATA_DIR = './data'


data = []
labels = []


for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)


    if not os.path.isdir(dir_path):
        continue

    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)


        if not os.path.isfile(img_path) or not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Unable to read {img_path}, skipping...")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)

            if data_aux:
                data.append(data_aux)
                labels.append(dir_)


with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)


