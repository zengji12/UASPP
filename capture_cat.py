import cv2
import os
import time
import streamlit as st

def create_folder_for_label(name, base_folder='data'):
    existing_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    label_folder = next((f for f in existing_folders if f.endswith(f".{name}")), None)

    if label_folder is None:
        id = len(existing_folders) + 1
        label_folder = os.path.join(base_folder, f"{id}.{name}")
        os.makedirs(label_folder, exist_ok=True)

    return label_folder

def detect_and_capture_faces(label, label_folder, capture_interval=1, break_button=None):
    face_cascade = cv2.CascadeClassifier("opencv-files/haarcascade_frontalface_alt.xml")

    last_capture_time = time.time()
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7, minSize=(70, 70))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            current_time = time.time()
            if current_time - last_capture_time >= capture_interval:
                captured_face = frame[y:y+h, x:x+w]
                timestamp = int(current_time)
                capture_path = os.path.join(label_folder, f"{label}_{timestamp}.jpg")
                cv2.imwrite(capture_path, captured_face)
                st.success(f"Captured face saved: {capture_path}")
                last_capture_time = current_time

        FRAME_WINDOW.image(frame, channels="BGR")
        
        if break_button:
            break

    FRAME_WINDOW.image([])
    cap.release()
    cv2.destroyAllWindows()
    
def capture_face():
    st.title("Face Capture Program")
    st.write("Face Detection Capture to Train Machine Learning")

    form = st.form(key='my-form')
    label = form.text_input('Enter the folder name for this label:')
    submit = form.form_submit_button('Submit')

    capture_folder='data'
    if len(label) > 0:
        label_folder = create_folder_for_label(label, capture_folder)

    break_button = st.empty()
    if label:
        stop_button = break_button.button("Stop")
        if not stop_button:
            detect_and_capture_faces(label, label_folder, break_button=stop_button)

        if stop_button:
            break_button.empty()
            
    st.image([])

    
