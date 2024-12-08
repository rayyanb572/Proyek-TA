import os
import shutil
import streamlit as st
import cv2
import pickle
import numpy as np
from keras_facenet import FaceNet
from ultralytics import YOLO
import zipfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load models and embeddings
@st.cache_resource
def load_yolo_model():
    logging.info("Loading YOLO model...")
    return YOLO("yolov8n-face.pt")

@st.cache_resource
def load_face_embeddings():
    logging.info("Loading face embeddings...")
    with open("face_embeddings.pkl", "rb") as f:
        return pickle.load(f)

# Initialize models
yolo_model = load_yolo_model()
known_embeddings = load_face_embeddings()
embedder = FaceNet()

# Utility functions
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_match(embedding, threshold=0.8):
    best_match = None
    best_score = threshold
    for person_name, person_embeddings in known_embeddings.items():
        for person_embedding in person_embeddings:
            score = cosine_similarity(embedding, person_embedding)
            if score > best_score:
                best_match = person_name
                best_score = score
    return best_match

def crop_face(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return image[y1:y2, x1:x2]

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

def classify_faces(file_list, output_folder="output_test"):
    # Create necessary folders
    os.makedirs("uploads", exist_ok=True)
    unknown_folder = os.path.join(output_folder, "unknown")
    clear_folder(output_folder)
    clear_folder(unknown_folder)

    # Process each uploaded file
    for file in file_list:
        try:
            # Save uploaded file
            temp_path = os.path.join("uploads", file.name)
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())

            # Read and validate image
            image = cv2.imread(temp_path)
            if image is None:
                st.warning(f"Invalid image: {file.name}")
                continue

            # Detect faces
            results = yolo_model.predict(image)
            if not results or not results[0].boxes:
                st.warning(f"No faces found in {file.name}")
                continue

            # Process each detected face
            for bbox in results[0].boxes.xyxy.numpy():
                face_image = crop_face(image, bbox)
                face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_embedding = embedder.embeddings([face_image_rgb])[0]

                # Find and categorize face
                match = find_match(face_embedding)
                if match:
                    person_folder = os.path.join(output_folder, match)
                    os.makedirs(person_folder, exist_ok=True)
                    shutil.copy(temp_path, person_folder)
                else:
                    shutil.copy(temp_path, unknown_folder)

        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    return output_folder

def zip_folder(folder_path, zip_name):
    zip_path = f"{zip_name}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, start=folder_path)
                zipf.write(file_path, arcname)
    return zip_path

def reset_application_state():
    # Clear physical folders
    folders_to_clear = ["uploads", "output_test"]
    for folder in folders_to_clear:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
    
    # Reset session states
    states_to_reset = [
        "uploaded_files", 
        "processing_completed", 
        "output_folder", 
        "total_files_uploaded"
    ]
    
    for state in states_to_reset:
        if state in st.session_state:
            del st.session_state[state]

def main():
    st.title("Face Classification App")
    st.write("Upload all image files you want to classify.")

    # Clear button
    if st.button("Clear All"):
        reset_application_state()
        st.rerun()

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Image Files", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )

    # Track uploaded files
    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) successfully uploaded.")

    # Process button
    if uploaded_files:
        if st.button("Process Images"):
            with st.spinner("Processing images..."):
                try:
                    output_folder = classify_faces(uploaded_files)
                    st.success(f"Processing complete! Output folder: {output_folder}")

                    # Display processed images
                    for folder_name in sorted(os.listdir(output_folder)):
                        folder_path = os.path.join(output_folder, folder_name)
                        st.write(f"\U0001F4C2 Folder: {folder_name}")

                        files = sorted(os.listdir(folder_path))
                        num_columns = 4
                        columns = st.columns(num_columns)

                        for idx, file_name in enumerate(files[:8]):
                            file_path = os.path.join(folder_path, file_name)
                            column_idx = idx % num_columns
                            with columns[column_idx]:
                                st.image(file_path, caption=file_name, width=150)

                    # Zip and download option
                    zip_path = zip_folder(output_folder, "output_test")
                    with open(zip_path, "rb") as zip_file:
                        st.download_button(
                            label="Download Processed Output",
                            data=zip_file,
                            file_name="output_test.zip",
                            mime="application/zip"
                        )

                except Exception as e:
                    st.error(f"Processing failed: {e}")

if __name__ == "__main__":
    main()