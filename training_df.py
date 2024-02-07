import cv2
import os
import numpy as np
from PIL import Image
import streamlit as st

def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y+w, x:x+h], faces[0]

def prepare_training_data(data_folder_path):
    faces = []
    labels = []

    for person_folder in os.listdir(data_folder_path):
        person_path = os.path.join(data_folder_path, person_folder)

        if os.path.isdir(person_path):
            label = int(person_folder.split('.')[0])  # Konversi label ke integer
            nama = person_folder.split('.')
            nama = nama[1].strip()
            st.success(f"Train Start for: {nama}")

            for file_name in os.listdir(person_path):
                image_path = os.path.join(person_path, file_name)

                # Check if the file is an image
                if image_path.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    image = cv2.imread(image_path)
                    face, rect = detect_face(image)
            
                    if face is not None:
                        faces.append(face)
                        labels.append(label)

    return faces, labels


def train_data():
    st.title("Train Data Set")
    st.write("Train the data with machine learning")

    the_button = st.empty()
    start_button = the_button.button("Start")
    if start_button:    
        data_folder_path = 'data/'
        faces, labels = prepare_training_data(data_folder_path)

        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(faces, np.array(labels, dtype=np.int32))
        face_recognizer.save('recognizer/train.yml')

        st.write("Data prepared")
