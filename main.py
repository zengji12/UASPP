import streamlit as st
from capture_cat import capture_face
from face_recog import face_recognite
from training_df import train_data
  
def main():
    button_width = 300

    # Apply CSS styling for button width
    st.markdown(
        f"""
        <style>
            div.stButton > button {{
                width: {button_width}px;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    titles = st.empty()
    nama = titles.title("Welcome to Face Recognition App!")
    desc = st.empty()
    descs = desc.write("This application allows you to capture faces and perform face recognition.")
        
    options = ["Home", "About", "Capture Faces", "Train Data", "Face Recognition"]
    choice = st.sidebar.selectbox("Select", options)


    if choice == "Capture Faces":
        titles.empty()
        desc.empty()
        capture_face()

    elif choice == "Face Recognition":
        titles.empty()
        desc.empty()
        face_recognite()

    elif choice == "Train Data":
        titles.empty()
        desc.empty()
        train_data()

    elif choice == "About":
        titles.empty()
        desc.empty()
        about_page()

    elif choice == "Home":
        home_page()

def home_page():
    st.empty()

def about_page():
    st.empty()
    st.title("About Face Recognition App")
    st.write("This application is designed for face detection, capture, and recognition.")
    st.write("Developed with Streamlit and OpenCV.")
    st.write("Author: Reza Ardiansyah Yudhanegara")

if __name__ == "__main__":
    main()
