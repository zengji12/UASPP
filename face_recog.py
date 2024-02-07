import cv2
import streamlit as st
import numpy as np
import os
import time
from skimage import filters
import pandas as pd
from datetime import datetime

def tandaAbsensi(name):
    #timestr = datetime.strftime("%Y%m%d-%H%M%S")
    now = datetime.now()
    dtstring = now.strftime("%H:%M:%S")
    datestr = time.strftime("%Y%m%d")
    with open( "absensi/" + datestr + '.csv','w+') as f:
        mydatalist = f.readlines()
        nameList = []
        for line in mydatalist:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            f.writelines(f'\n{name},{dtstring}')
            
def texture_analysis(face_image):
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    sobel_energy = filters.sobel(gray_face)
    mean_energy = np.mean(sobel_energy)
    threshold = 0.8 * mean_energy
    return np.mean(sobel_energy) > threshold

def face_recognition_program(break_button=None): 
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    recognizer.read('recognizer/train.yml')

    face_cascade = cv2.CascadeClassifier("opencv-files/haarcascade_frontalface_alt.xml")

    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0

    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    
    data_folder_path = 'data/'
    known_names = []

    for folder_name in os.listdir(data_folder_path):
        name = folder_name.split('.')[1] if '.' in folder_name else folder_name
        known_names.append(name)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]

            is_real_face = texture_analysis(frame[y:y+h, x:x+w])

            if is_real_face:
                label, confidence = recognizer.predict(face_img)

                if confidence < 50:
                    nama = known_names[id]
                    label_text = f"Matched: {nama}"
                    color = (0, 255, 0)
                else:
                    label_text = "Unknown"
                    color = (0, 0, 255)
            else:
                label_text = "Fake Face (Detected Texture), Photo"
                color = (0, 0, 255)

            cv2.putText(frame, label_text, (x, y-5), font, 0.5, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)


        FRAME_WINDOW.image(frame, channels="BGR")
        tandaAbsensi(known_names[id-1])
        if break_button:
            break

    FRAME_WINDOW.image([])
    cap.release()
    cv2.destroyAllWindows()

def face_recognite():
    st.title("Face Recognition Program")
    start_button = st.button("Start")

    break_button = st.empty()    
    if start_button:
        stop_button = break_button.button("Stop")
        if not stop_button:
            face_recognition_program(break_button=stop_button)
            
        if stop_button:
            break_button.empty()

    st.image([])
