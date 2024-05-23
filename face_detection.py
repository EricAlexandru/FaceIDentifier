import cv2
import dlib
import os
import numpy as np

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Load known faces from the "profiles" folder
profiles_folder = "profiles"
known_faces = {}
for profile_name in os.listdir(profiles_folder):
    profile_folder = os.path.join(profiles_folder, profile_name)
    if os.path.isdir(profile_folder):
        known_faces[profile_name] = load_images_from_folder(profile_folder)

# Initialize face detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Capture video from port 1
cap = cv2.VideoCapture(0)

# Reduce frame resolution
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Initialize variables for face recognition interval and frame count
recognition_interval = 5  # Perform recognition every 5 frames
frame_count = 0

# Initialize an ID counter for detected faces
face_id_counter = 0

# Initialize an empty dictionary to track detected faces
detected_faces = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % recognition_interval != 0:
        continue

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Iterate over detected faces
    for face in faces:
        # Generate a unique ID for the face
        face_id = face_id_counter
        face_id_counter += 1

        # Track detected faces
        detected_faces[face_id] = face

        # Perform recognition for tracked faces
        if face_id in detected_faces:
            # Get the facial landmarks
            shape = predictor(gray, face)
            face_descriptor = face_recognizer.compute_face_descriptor(frame, shape)

            # Compare the face descriptor with known faces
            match_found = False
            for name, face_images in known_faces.items():
                for known_face_image in face_images:
                    known_face_descriptor = face_recognizer.compute_face_descriptor(known_face_image, shape)
                    # Compare the descriptors
                    similarity = np.linalg.norm(np.array(face_descriptor) - np.array(known_face_descriptor))
                    if similarity < 0.6:  # Adjust the threshold as needed
                        match_found = True
                        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
                        text = f"Match: {name}"
                        break
                if match_found:
                    break
            
            # If no match found, draw rectangle in red
            if not match_found:
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
                text = "Unknown"

            # Draw text at the bottom right of the rectangle
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = face.right() - text_size[0]
            text_y = face.bottom() + text_size[1] + 10
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if match_found else (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
