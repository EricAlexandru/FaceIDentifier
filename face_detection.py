import cv2
import numpy as np
import dlib
import sqlite3
import threading
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLineEdit, QListWidget, QMessageBox, QFrame, QGridLayout, QListWidgetItem)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image
import sys
import time

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained face landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the pre-trained face recognition model
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Connect to SQLite database
conn = sqlite3.connect('faces.db')
c = conn.cursor()

# Ensure presence table exists
c.execute('''
CREATE TABLE IF NOT EXISTS presence (
    profile_id INTEGER PRIMARY KEY,
    is_present BOOLEAN NOT NULL DEFAULT 0,
    FOREIGN KEY(profile_id) REFERENCES profiles(id)
)
''')
conn.commit()

# Ensure descriptors table exists
c.execute('''
CREATE TABLE IF NOT EXISTS descriptors (
    profile_id INTEGER,
    descriptor BLOB,
    FOREIGN KEY(profile_id) REFERENCES profiles(id)
)
''')
conn.commit()

# Dictionary to store known faces
known_faces = {}

def update_known_faces():
    global known_faces
    known_faces.clear()
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    c.execute("SELECT profiles.id, profiles.name, presence.is_present FROM profiles JOIN presence ON profiles.id = presence.profile_id")
    profiles = c.fetchall()
    for profile_id, profile_name, is_present in profiles:
        c.execute("SELECT descriptor FROM descriptors WHERE profile_id=?", (profile_id,))
        descriptors = [np.frombuffer(row[0], dtype=np.float64) for row in c.fetchall()]
        known_faces[profile_name] = (descriptors, is_present)
    conn.close()

# Initialize known faces
update_known_faces()

# Initialize the video capture objects
entrance_cap = cv2.VideoCapture(0)
exit_cap = cv2.VideoCapture(1)  # Assuming the second camera is used for exit

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Recognition and Presence Verification")
        self.setStyleSheet("""
            QWidget {
                background-color: #2c3e50;
                color: #ecf0f1;
            }
            QPushButton {
                background-color: #2980b9;
                color: #ecf0f1;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QLineEdit, QListWidget {
                background-color: #34495e;
                color: #ecf0f1;
                border: none;
                padding: 5px;
            }
            QLabel {
                font-size: 14px;
                font-weight: bold;
            }
        """)

        self.entrance_video_label = QLabel(self)
        self.exit_video_label = QLabel(self)

        self.entry_profile_name = QLineEdit(self)
        self.entry_profile_name.setPlaceholderText("Enter profile name")
        self.create_profile_button = QPushButton("Create Profile", self)
        self.create_profile_button.clicked.connect(self.save_profile)

        self.selected_folder_label = QLabel("Profiles", self)
        self.selected_folder = QListWidget(self)
        self.update_profile_list()

        self.search_var = QLineEdit(self)
        self.search_var.setPlaceholderText("Search profiles")
        self.search_var.textChanged.connect(self.search_profiles)

        self.capture_photo_button = QPushButton("Capture Photo", self)
        self.capture_photo_button.clicked.connect(self.capture_image)

        self.delete_profile_button = QPushButton("Delete Profile", self)
        self.delete_profile_button.clicked.connect(self.delete_profile)

        self.database_button = QPushButton("Open Database", self)
        self.database_button.clicked.connect(self.open_database_view)

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close)

        self.presence_list_label = QLabel("Present Profiles", self)
        self.presence_list = QListWidget(self)

        self.search_presence_var = QLineEdit(self)
        self.search_presence_var.setPlaceholderText("Search presence")
        self.search_presence_var.textChanged.connect(self.search_presence)

        self.database_widget = QWidget(self)
        self.database_widget.setStyleSheet("""
            QWidget {
                background-color: #34495e;
            }
        """)
        self.database_widget.hide()

        self.profile_list_widget = QListWidget(self.database_widget)
        self.profile_search_bar = QLineEdit(self.database_widget)
        self.profile_search_bar.setPlaceholderText("Search profiles")
        self.profile_search_bar.textChanged.connect(self.search_profiles_in_db)

        self.profile_picture_label = QLabel(self.database_widget)
        self.profile_picture_label.setFixedSize(200, 200)
        self.profile_picture_label.setStyleSheet("border: 1px solid #ecf0f1;")

        self.db_delete_profile_button = QPushButton("Delete Profile", self.database_widget)
        self.db_delete_profile_button.clicked.connect(self.delete_profile_in_db)

        db_layout = QVBoxLayout(self.database_widget)
        db_layout.addWidget(self.profile_search_bar)
        db_layout.addWidget(self.profile_list_widget)
        db_layout.addWidget(self.profile_picture_label)
        db_layout.addWidget(self.db_delete_profile_button)
        self.database_widget.setLayout(db_layout)

        self.profile_list_widget.itemSelectionChanged.connect(self.display_profile_picture)

        video_layout = QHBoxLayout()
        video_layout.addWidget(self.entrance_video_label)
        video_layout.addWidget(self.exit_video_label)

        profile_control_layout = QHBoxLayout()
        profile_control_layout.addWidget(self.entry_profile_name)
        profile_control_layout.addWidget(self.create_profile_button)

        profile_list_layout = QVBoxLayout()
        profile_list_layout.addWidget(self.selected_folder_label)
        profile_list_layout.addWidget(self.search_var)
        profile_list_layout.addWidget(self.selected_folder)
        profile_list_layout.addWidget(self.capture_photo_button)
        profile_list_layout.addWidget(self.delete_profile_button)
        profile_list_layout.addWidget(self.database_button)

        presence_list_layout = QVBoxLayout()
        presence_list_layout.addWidget(self.presence_list_label)
        presence_list_layout.addWidget(self.search_presence_var)
        presence_list_layout.addWidget(self.presence_list)

        main_layout = QGridLayout()
        main_layout.addLayout(video_layout, 0, 0, 1, 2)
        main_layout.addLayout(profile_control_layout, 1, 0, 1, 2)
        main_layout.addLayout(profile_list_layout, 2, 0)
        main_layout.addLayout(presence_list_layout, 2, 1)
        main_layout.addWidget(self.exit_button, 3, 0, 1, 2, alignment=Qt.AlignCenter)
        main_layout.addWidget(self.database_widget, 0, 2, 4, 1)

        self.setLayout(main_layout)

        self.entrance_timer = QTimer()
        self.entrance_timer.timeout.connect(self.update_entrance_video_feed)
        self.entrance_timer.start(30)

        self.exit_timer = QTimer()
        self.exit_timer.timeout.connect(self.update_exit_video_feed)
        self.exit_timer.start(30)

        self.presence_timer = QTimer()
        self.presence_timer.timeout.connect(self.update_presence_list)
        self.presence_timer.start(10000)

    def update_entrance_video_feed(self):
        ret, frame = entrance_cap.read()
        if ret:
            self.process_frame(frame, "entrance")
            self.display_frame(frame, self.entrance_video_label)

    def update_exit_video_feed(self):
        ret, frame = exit_cap.read()
        if ret:
            self.process_frame(frame, "exit")
            self.display_frame(frame, self.exit_video_label)

    def process_frame(self, frame, camera_type):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        for (x, y, w, h) in faces:
            shape = predictor(gray, dlib.rectangle(x, y, x + w, y + h))
            descriptor = face_recognizer.compute_face_descriptor(frame, shape)
            match_found = False
            for name, (descriptors, is_present) in known_faces.items():
                for known_descriptor in descriptors:
                    similarity = np.linalg.norm(np.array(descriptor) - np.array(known_descriptor))
                    if similarity < 0.6:  # Adjust the threshold as needed
                        if camera_type == "entrance" and not is_present:
                            c.execute("UPDATE presence SET is_present = 1 WHERE profile_id = (SELECT id FROM profiles WHERE name=?)", (name,))
                            conn.commit()
                        elif camera_type == "exit" and is_present:
                            c.execute("UPDATE presence SET is_present = 0 WHERE profile_id = (SELECT id FROM profiles WHERE name=?)", (name,))
                            conn.commit()
                        match_found = True
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0) if camera_type == "entrance" else (0, 0, 255), 2)
                        cv2.putText(frame, f"{name} {'(Entered)' if camera_type == 'entrance' else '(Exited)'}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36, 255, 12), 2)
                        break
                if match_found:
                    break
            if not match_found:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        conn.close()

    def display_frame(self, frame, label):
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        label.setPixmap(QPixmap.fromImage(qImg))

    def save_profile(self):
        profile_name = self.entry_profile_name.text()
        if profile_name:
            conn = sqlite3.connect('faces.db')
            c = conn.cursor()
            c.execute("INSERT OR IGNORE INTO profiles (name) VALUES (?)", (profile_name,))
            conn.commit()
            c.execute("SELECT id FROM profiles WHERE name=?", (profile_name,))
            profile_id = c.fetchone()[0]
            c.execute("INSERT INTO presence (profile_id) VALUES (?)", (profile_id,))
            conn.commit()
            conn.close()
            update_known_faces()
            self.update_profile_list()
            print(f"Profile '{profile_name}' created successfully.")
        else:
            print("Please enter a profile name.")

    def update_profile_list(self):
        self.selected_folder.clear()
        for folder_name in sorted(known_faces.keys()):
            self.selected_folder.addItem(folder_name)

    def search_profiles(self, text):
        search_term = text.lower()
        self.selected_folder.clear()
        matches = [folder_name for folder_name in sorted(known_faces.keys()) if folder_name.lower().startswith(search_term)]
        self.selected_folder.addItems(matches) if matches else self.selected_folder.addItem("No match")

    def delete_profile(self):
        selected_items = self.selected_folder.selectedItems()
        if selected_items:
            folder_name = selected_items[0].text()
            conn = sqlite3.connect('faces.db')
            c = conn.cursor()
            c.execute("SELECT id FROM profiles WHERE name=?", (folder_name,))
            profile_id = c.fetchone()[0]
            c.execute("DELETE FROM descriptors WHERE profile_id=?", (profile_id,))
            c.execute("DELETE FROM profiles WHERE id=?", (profile_id,))
            c.execute("DELETE FROM presence WHERE profile_id=?", (profile_id,))
            conn.commit()
            conn.close()
            update_known_faces()
            self.update_profile_list()
            print(f"Profile '{folder_name}' deleted successfully.")
        else:
            print("Please select a profile to delete.")

    def capture_image(self):
        selected_items = self.selected_folder.selectedItems()
        if selected_items:
            folder_name = selected_items[0].text()
            conn = sqlite3.connect('faces.db')
            c = conn.cursor()
            c.execute("SELECT id FROM profiles WHERE name=?", (folder_name,))
            profile_id = c.fetchone()[0]
            ret, frame = entrance_cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                for (x, y, w, h) in faces:
                    shape = predictor(gray, dlib.rectangle(x, y, x + w, y + h))
                    descriptor = np.array(face_recognizer.compute_face_descriptor(frame, shape), dtype=np.float64)
                    c.execute("INSERT INTO descriptors (profile_id, descriptor) VALUES (?, ?)", (profile_id, descriptor.tobytes()))
                    conn.commit()
                    conn.close()
                    update_known_faces()
                    print(f"Image captured and saved for profile '{folder_name}'.")
                    return
                print("No face detected.")
            else:
                print("Failed to capture image from camera.")
        else:
            print("Please select a folder before capturing a photo.")

    def update_presence_list(self):
        self.presence_list.clear()
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        c.execute("SELECT profiles.name FROM profiles JOIN presence ON profiles.id = presence.profile_id WHERE presence.is_present = 1")
        present_profiles = c.fetchall()
        conn.close()
        for profile_name in present_profiles:
            self.presence_list.addItem(profile_name[0])

    def open_database_view(self):
        self.database_widget.show()
        self.update_profile_list_in_db()

    def update_profile_list_in_db(self):
        self.profile_list_widget.clear()
        for folder_name in sorted(known_faces.keys()):
            item = QListWidgetItem(folder_name)
            self.profile_list_widget.addItem(item)

    def search_profiles_in_db(self, text):
        search_term = text.lower()
        self.profile_list_widget.clear()
        matches = [folder_name for folder_name in sorted(known_faces.keys()) if folder_name.lower().startswith(search_term)]
        self.profile_list_widget.addItems(matches) if matches else self.profile_list_widget.addItem("No match")

    def display_profile_picture(self):
        selected_item = self.profile_list_widget.currentItem()
        if selected_item:
            folder_name = selected_item.text()
            conn = sqlite3.connect('faces.db')
            c = conn.cursor()
            c.execute("SELECT id FROM profiles WHERE name=?", (folder_name,))
            profile_id = c.fetchone()[0]
            c.execute("SELECT descriptor FROM descriptors WHERE profile_id=?", (profile_id,))
            descriptor = np.frombuffer(c.fetchone()[0], dtype=np.float64)
            profile_picture = np.zeros((200, 200, 3), dtype=np.uint8)
            shape = predictor(profile_picture, dlib.rectangle(0, 0, 200, 200))
            profile_picture = cv2.cvtColor(profile_picture, cv2.COLOR_RGB2BGR)
            cv2.putText(profile_picture, folder_name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(profile_picture, (10, 30), (190, 190), (255, 255, 255), 1)
            for i in range(68):
                cv2.circle(profile_picture, (int(shape.part(i).x), int(shape.part(i).y)), 1, (255, 255, 255), 1)
            cv2.imshow("Profile Picture", profile_picture)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def delete_profile_in_db(self):
        selected_item = self.profile_list_widget.currentItem()
        if selected_item:
            folder_name = selected_item.text()
            conn = sqlite3.connect('faces.db')
            c = conn.cursor()
            c.execute("SELECT id FROM profiles WHERE name=?", (folder_name,))
            profile_id = c.fetchone()[0]
            c.execute("DELETE FROM descriptors WHERE profile_id=?", (profile_id,))
            c.execute("DELETE FROM profiles WHERE id=?", (profile_id,))
            c.execute("DELETE FROM presence WHERE profile_id=?", (profile_id,))
            conn.commit()
            conn.close()
            update_known_faces()
            self.update_profile_list_in_db()
            print(f"Profile '{folder_name}' deleted successfully.")
        else:
            print("Please select a profile to delete.")

    def search_presence(self, text):
        search_term = text.lower()
        self.presence_list.clear()
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        c.execute("SELECT profiles.name FROM profiles JOIN presence ON profiles.id = presence.profile_id WHERE presence.is_present = 1")
        present_profiles = c.fetchall()
        conn.close()
        matches = [profile_name[0] for profile_name in present_profiles if profile_name[0].lower().startswith(search_term)]
        self.presence_list.addItems(matches) if matches else self.presence_list.addItem("No match")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

