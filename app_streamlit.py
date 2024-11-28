import os
import shutil
import streamlit as st
import cv2
import pickle
import numpy as np
from keras_facenet import FaceNet
from ultralytics import YOLO

# --- Load YOLO Model and Face Embeddings ---
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n-face.pt")

@st.cache_resource
def load_face_embeddings():
    with open("face_embeddings.pkl", "rb") as f:
        return pickle.load(f)

yolo_model = load_yolo_model()
known_embeddings = load_face_embeddings()
embedder = FaceNet()

# --- Utility Functions ---
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
    unknown_folder = os.path.join(output_folder, "unknown")
    clear_folder(output_folder)
    clear_folder(unknown_folder)

    for file in file_list:
        temp_path = os.path.join("uploads", file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        image = cv2.imread(temp_path)

        results = yolo_model.predict(image)
        if not results or not results[0].boxes:
            continue

        for bbox in results[0].boxes.xyxy.numpy():
            face_image = crop_face(image, bbox)
            face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_embedding = embedder.embeddings([face_image_rgb])[0]

            match = find_match(face_embedding)
            if match:
                person_folder = os.path.join(output_folder, match)
                os.makedirs(person_folder, exist_ok=True)
                shutil.copy(temp_path, person_folder)
            else:
                shutil.copy(temp_path, unknown_folder)

    return output_folder

# --- Streamlit Application ---
def clear_uploads_folder(folder_path="uploads"):
    """
    Clear the contents of the uploads folder.
    """
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
            except Exception as e:
                print(f"Error while deleting {file_path}: {e}")
    os.makedirs(folder_path, exist_ok=True)

def main():
    st.title("Face Classification App")
    st.write("Unggah semua file gambar yang ingin diklasifikasikan.")

    # --- Tombol untuk Menghapus Isi Folder Uploads ---
    if st.button("Hapus Semua Isi Uploads"):
        clear_uploads_folder()
        st.success("Isi folder 'uploads' telah dihapus.")

    # --- Pengunggahan File ---
    uploaded_files = st.file_uploader("Upload File Gambar", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        st.write(f"{len(uploaded_files)} file berhasil diunggah.")
        if st.button("Proses Gambar"):
            output_folder = classify_faces(uploaded_files)
            st.success(f"Proses selesai! Folder output: {output_folder}")

            if os.path.exists(output_folder):
                for folder_name in os.listdir(output_folder):
                    folder_path = os.path.join(output_folder, folder_name)
                    st.write(f"📂 Folder: {folder_name}")
                    for file_name in os.listdir(folder_path):
                        st.image(os.path.join(folder_path, file_name), caption=file_name, use_column_width=True)

if __name__ == "__main__":
    main()
