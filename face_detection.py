import cv2
import os
import numpy as np
import dlib

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained face landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the pre-trained face recognition model
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Function to compute face descriptors
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

# Function to create a new profile
def create_profile(profile_name):
    os.makedirs(os.path.join("profiles", profile_name))

# Function to add pictures to a profile
def add_picture_to_profile(profile_name, picture):
    profile_folder = os.path.join("profiles", profile_name)
    cv2.imwrite(os.path.join(profile_folder, f"{len(os.listdir(profile_folder)) + 1}.jpg"), picture)

# Function to get all available profiles
def get_available_profiles():
    return [profile_name for profile_name in os.listdir("profiles") if os.path.isdir(os.path.join("profiles", profile_name))]

# Load known faces from the "profiles" folder
known_faces = {}
for profile_name in get_available_profiles():
    profile_folder = os.path.join("profiles", profile_name)
    images = load_images_from_folder(profile_folder)
    descriptors = compute_face_descriptors(images)
    known_faces[profile_name] = descriptors

# Initialize variables for profile selection and new profile creation
selected_profile = None
new_profile_name = ""

# Capture video from port 0
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]

        # Compute descriptor for the detected face
        shape = predictor(gray, dlib.rectangle(x, y, x+w, y+h))
        descriptor = face_recognizer.compute_face_descriptor(frame, shape)

        # Compare with known faces
        match_found = False
        for name, descriptors in known_faces.items():
            for known_descriptor in descriptors:
                similarity = np.linalg.norm(np.array(descriptor) - np.array(known_descriptor))
                if similarity < 0.6:  # Adjust the threshold as needed
                    match_found = True
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36,255,12), 2)
                    break
            if match_found:
                break

        # If no match found, draw rectangle in red
        if not match_found:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    # Display available profiles
    profiles = get_available_profiles()
    profile_text = "Available Profiles: " + ", ".join(profiles)
    cv2.putText(frame, profile_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display selected profile
    if selected_profile:
        cv2.putText(frame, "Selected Profile: " + selected_profile, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display new profile creation input
    if new_profile_name:
        cv2.putText(frame, "Creating Profile: " + new_profile_name, (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Check for key press events
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):  # Press 'c' to capture image for new profile or selected profile
        if new_profile_name:
            create_profile(new_profile_name)
            known_faces[new_profile_name] = compute_face_descriptors([frame])
            new_profile_name = ""
        elif selected_profile:
            add_picture_to_profile(selected_profile, frame)
    elif key == ord('n'):  # Press 'n' to initiate new profile creation
        new_profile_name = input("Enter profile name: ")
    elif key == ord('s'):  # Press 's' to select a profile
        selected_profile = input("Select profile: ")

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
