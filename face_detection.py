import cv2
import os
import numpy as np
import dlib
from tkinter import Tk, Button, Label, Listbox, Entry, END, Scrollbar, Frame, BOTH, RIGHT, LEFT, Y, StringVar
from PIL import Image, ImageTk
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def compute_face_descriptors(images):
    descriptors = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            shape = predictor(gray, dlib.rectangle(x, y, x+w, y+h))
            descriptor = face_recognizer.compute_face_descriptor(img, shape)
            descriptors.append(descriptor)
    return descriptors

def save_image(image, folder, filename):
    cv2.imwrite(os.path.join(folder, filename), image)

def update_known_faces():
    global known_faces
    for profile_name in sorted(os.listdir(profiles_folder)):
        profile_folder = os.path.join(profiles_folder, profile_name)
        if os.path.isdir(profile_folder) and profile_name not in known_faces:
            images = load_images_from_folder(profile_folder)
            descriptors = compute_face_descriptors(images)
            known_faces[profile_name] = descriptors

profiles_folder = "profiles"
known_faces = {}
update_known_faces()

cap = cv2.VideoCapture(0)
root = Tk()
root.title("Face Recognition and Photo Capture")
video_label = Label(root)
video_label.pack()
entry_profile_name = Entry(root)
create_profile_button = None
last_recognized_face = None

def update_video_feed():
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            shape = predictor(gray, dlib.rectangle(x, y, x+w, y+h))
            descriptor = face_recognizer.compute_face_descriptor(frame, shape)
            match_found = False
            for name, descriptors in known_faces.items():
                for known_descriptor in descriptors:
                    similarity = np.linalg.norm(np.array(descriptor) - np.array(known_descriptor))
                    if similarity < 0.6:
                        match_found = True
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,255,12), 2)
                        last_recognized_face = name
                        break
                if match_found:
                    break
            if not match_found:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        ratio = frame.shape[1] / frame.shape[0]
        new_width = 640
        new_height = int(new_width / ratio)
        frame = cv2.resize(frame, (new_width, new_height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    video_label.after(10, update_video_feed)

update_video_feed()

def create_profile():
    global create_profile_button
    global entry_profile_name
    if not create_profile_button:
        create_profile_button = Button(root, text="Create", command=save_profile)
        create_profile_button.pack()
        entry_profile_name.pack()

def save_profile():
    profile_name = entry_profile_name.get()
    if profile_name:
        if not os.path.exists(os.path.join(profiles_folder, profile_name)):
            os.makedirs(os.path.join(profiles_folder, profile_name))
            update_known_faces()
            selected_folder.insert(END, profile_name)
            print(f"Profile '{profile_name}' created successfully.")
        else:
            print(f"Profile '{profile_name}' already exists.")
    else:
        print("Please enter a profile name.")

selected_folder_frame = Frame(root)
selected_folder_scrollbar = Scrollbar(selected_folder_frame)
selected_folder_scrollbar.pack(side=RIGHT, fill=Y)
selected_folder = Listbox(selected_folder_frame, selectmode="single", height=5, yscrollcommand=selected_folder_scrollbar.set)
for folder_name in sorted(known_faces.keys()):
    selected_folder.insert(END, folder_name)
selected_folder.pack(side=LEFT, fill=BOTH)
selected_folder_scrollbar.config(command=selected_folder.yview)
selected_folder_frame.pack()

search_var = StringVar()
search_var.trace("w", lambda name, index, mode, sv=search_var: search_profiles(sv))
search_entry = Entry(root, textvariable=search_var)
search_entry.pack()

def search_profiles(search_var):
    search_term = search_var.get().lower()
    selected_folder.delete(0, END)
    matches = [folder_name for folder_name in sorted(known_faces.keys()) if folder_name.lower().startswith(search_term)]
    for folder_name in matches:
        selected_folder.insert(END, folder_name)
    if not matches:
        selected_folder.insert(END, "No match")

def delete_profile():
    if selected_folder.curselection():
        index = selected_folder.curselection()[0]
        folder_name = selected_folder.get(index)
        profile_path = os.path.join(profiles_folder, folder_name)
        if os.path.exists(profile_path):
            import shutil
            shutil.rmtree(profile_path)
            del known_faces[folder_name]
            selected_folder.delete(index)
            print(f"Profile '{folder_name}' deleted successfully.")
        else:
            print(f"Profile '{folder_name}' does not exist.")
    else:
        print("Please select a profile to delete.")

Button(root, text="Delete Profile", command=delete_profile).pack()

def capture_image():
    if selected_folder.curselection():
        index = selected_folder.curselection()[0]
        folder_name = selected_folder.get(index)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                shape = predictor(gray, dlib.rectangle(x, y, x+w, y+h))
                descriptor = face_recognizer.compute_face_descriptor(frame, shape)
                match_found = False
                for name, descriptors in known_faces.items():
                    for known_descriptor in descriptors:
                        similarity = np.linalg.norm(np.array(descriptor) - np.array(known_descriptor))
                        if similarity < 0.6:
                            match_found = True
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,255,12), 2)
                            break
                    if match_found:
                        break
                if not match_found:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            save_image(frame, os.path.join(profiles_folder, folder_name), f"image_{len(os.listdir(os.path.join(profiles_folder, folder_name))) + 1}.jpg")
    else:
        print("Please select a folder before capturing a photo.")

Button(root, text="Capture Photo", command=capture_image).pack()
Button(root, text="Exit", command=root.quit).pack()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
