import cv2
import numpy as np
import dlib
import sqlite3
import threading
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLineEdit, QListWidget, QMessageBox, QFrame, QGridLayout)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PIL import Image
import sys
import time

# Încarcă clasificatorul Haar pentru detectarea feței
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Încarcă predictorul de puncte caracteristice pentru față (68 de puncte)
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
# Încarcă modelul de recunoaștere facială
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Conectează la baza de date SQLite
conn = sqlite3.connect('faces.db')
c = conn.cursor()

# Creează tabelul de prezență dacă nu există deja
c.execute('''
CREATE TABLE IF NOT EXISTS presence (
    profile_id INTEGER PRIMARY KEY,
    is_present BOOLEAN NOT NULL DEFAULT 0,
    timestamp TEXT,
    FOREIGN KEY(profile_id) REFERENCES profiles(id)
)
''')
conn.commit()

# Dicționar pentru stocarea fețelor cunoscute
known_faces = {}

# Funcție pentru actualizarea fețelor cunoscute din baza de date
def update_known_faces():
    global known_faces
    known_faces.clear()
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    c.execute("SELECT profiles.id, profiles.name, presence.is_present, presence.timestamp FROM profiles JOIN presence ON profiles.id = presence.profile_id")
    profiles = c.fetchall()
    for profile_id, profile_name, is_present, timestamp in profiles:
        c.execute("SELECT descriptor FROM descriptors WHERE profile_id=?", (profile_id,))
        descriptors = [np.frombuffer(row[0], dtype=np.float64) for row in c.fetchall()]
        known_faces[profile_name] = (descriptors, is_present, timestamp)
    conn.close()

update_known_faces()

# Inițializează camerele video pentru intrare și ieșire
entrance_cap = cv2.VideoCapture(0)
entrance_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
entrance_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Se setează camera pentru a utiliza un dispozitiv specific (ID-ul 1)
exit_cap = cv2.VideoCapture(1)

# Setează lățimea și înălțimea imaginii capturate
# Aici am setat rezoluția la 1280x720, ceea ce este un HD standard
exit_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
exit_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Se ajustează expunerea camerei. Valoarea poate varia între -10 și 0
# -4 înseamnă o expunere ușor crescută, pentru a îmbunătăți luminozitatea
exit_cap.set(cv2.CAP_PROP_EXPOSURE, -4)

# Setează câștigul imaginii. Valoarea 0 înseamnă că nu amplificăm semnalul,
# iar orice valoare mai mare va amplifica semnalul și poate introduce zgomot
exit_cap.set(cv2.CAP_PROP_GAIN, 0)

# Setează contrastul imaginii. Valoarea 0 este echilibrată, iar orice valoare mai mare
# va face contrastul mai mare (mai multă diferență între culori deschise și întunecate)
exit_cap.set(cv2.CAP_PROP_CONTRAST, 0.5)

# Se ajustează saturația culorilor imaginii. Saturația mai mare înseamnă culori mai intense
exit_cap.set(cv2.CAP_PROP_SATURATION, 0.5)

# Setează claritatea imaginii. O valoare mai mare înseamnă o imagine mai clară, dar poate
# introduce și mai mult zgomot dacă valoarea este prea mare
exit_cap.set(cv2.CAP_PROP_SHARPNESS, 0.5)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Recognition and Presence Verification")
        # Stilizează interfața
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

        # Inițializează etichetele pentru afișarea fluxului video
        self.entrance_video_label = QLabel(self)
        self.exit_video_label = QLabel(self)

        # Inițializează câmpul pentru introducerea numelui profilului și butonul de creare profil
        self.entry_profile_name = QLineEdit(self)
        self.entry_profile_name.setPlaceholderText("Enter profile name")
        self.create_profile_button = QPushButton("Create Profile", self)
        self.create_profile_button.clicked.connect(self.save_profile)

        # Inițializează lista de profiluri și butoanele de control
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

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close)

        # Inițializează lista de prezențe și câmpul de căutare
        self.presence_list_label = QLabel("Present Profiles", self)
        self.presence_list = QListWidget(self)

        self.search_presence_var = QLineEdit(self)
        self.search_presence_var.setPlaceholderText("Search presence")
        self.search_presence_var.textChanged.connect(self.search_presence)

        # Aranjează componentele în layout-uri
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

        self.setLayout(main_layout)

        # Inițializează timer-ele pentru actualizarea fluxului video și a listei de prezențe
        self.entrance_timer = QTimer()
        self.entrance_timer.timeout.connect(self.update_entrance_video_feed)
        self.entrance_timer.start(30)

        self.exit_timer = QTimer()
        self.exit_timer.timeout.connect(self.update_exit_video_feed)
        self.exit_timer.start(30)

        self.presence_timer = QTimer()
        self.presence_timer.timeout.connect(self.update_presence_list)
        self.presence_timer.start(10000)

    # Funcție pentru actualizarea fluxului video de la intrare
    def update_entrance_video_feed(self):
        ret, frame = entrance_cap.read()
        if ret:
            self.process_frame(frame, "entrance")
            self.display_frame(frame, self.entrance_video_label)

    # Funcție pentru actualizarea fluxului video de la ieșire
    def update_exit_video_feed(self):
        ret, frame = exit_cap.read()
        if ret:
            self.process_frame(frame, "exit")
            self.display_frame(frame, self.exit_video_label)

    # Funcție pentru procesarea unui cadru video
    def process_frame(self, frame, camera_type):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        for (x, y, w, h) in faces:
            shape = predictor(gray, dlib.rectangle(x, y, x + w, y + h))
            descriptor = face_recognizer.compute_face_descriptor(frame, shape)
            descriptor_np = np.array(descriptor)
            match_found = False
            for name, (descriptors, is_present, timestamp) in known_faces.items():
                for known_descriptor in descriptors:
                    similarity = np.linalg.norm(descriptor_np - known_descriptor)
                    if similarity < 0.6:
                        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
                        if camera_type == "entrance" and not is_present:
                            c.execute("UPDATE presence SET is_present = 1, timestamp = ? WHERE profile_id = (SELECT id FROM profiles WHERE name=?)", (current_time, name))
                            conn.commit()
                            update_known_faces()
                        elif camera_type == "exit" and is_present:
                            c.execute("UPDATE presence SET is_present = 0, timestamp = ? WHERE profile_id = (SELECT id FROM profiles WHERE name=?)", (current_time, name))
                            conn.commit()
                            update_known_faces()
                        match_found = True
                        color = (0, 255, 0) if camera_type == "entrance" else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, f"{name} {'Entered' if camera_type == 'entrance' else 'Exited'}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (36, 255, 12), 2)
                        break
                if match_found:
                    break
            if not match_found:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        conn.close()

    # Funcție pentru afișarea unui cadru video într-un QLabel
    def display_frame(self, frame, label):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        qimg = QImage(frame.data, width, height, width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap)

    # Funcție pentru salvarea unui profil nou
    def save_profile(self):
        name = self.entry_profile_name.text()
        if name:
            conn = sqlite3.connect('faces.db')
            c = conn.cursor()
            c.execute("INSERT INTO profiles (name) VALUES (?)", (name,))
            profile_id = c.lastrowid
            c.execute("INSERT INTO presence (profile_id) VALUES (?)", (profile_id,))
            conn.commit()
            conn.close()
            self.entry_profile_name.clear()
            self.update_profile_list()
        else:
            QMessageBox.warning(self, "Input Error", "Please enter a profile name.")

    # Funcție pentru actualizarea listei de profiluri
    def update_profile_list(self):
        self.selected_folder.clear()
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        c.execute("SELECT name FROM profiles")
        profiles = c.fetchall()
        for profile in profiles:
            self.selected_folder.addItem(profile[0])
        conn.close()

    # Funcție pentru capturarea unei imagini pentru un profil selectat
    def capture_image(self):
        selected_profile = self.selected_folder.currentItem()
        if selected_profile:
            name = selected_profile.text()
            ret, frame = entrance_cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                if len(faces) == 1:
                    (x, y, w, h) = faces[0]
                    shape = predictor(gray, dlib.rectangle(x, y, x + w, y + h))
                    descriptor = face_recognizer.compute_face_descriptor(frame, shape)
                    descriptor_np = np.array(descriptor)
                    conn = sqlite3.connect('faces.db')
                    c = conn.cursor()
                    c.execute("SELECT id FROM profiles WHERE name = ?", (name,))
                    profile_id = c.fetchone()[0]
                    c.execute("INSERT INTO descriptors (profile_id, descriptor) VALUES (?, ?)", (profile_id, descriptor_np.tobytes()))
                    conn.commit()
                    conn.close()
                    QMessageBox.information(self, "Success", "Profile updated successfully.")
                    update_known_faces()
                else:
                    QMessageBox.warning(self, "Capture Error", "Could not detect a single face. Please try again.")
        else:
            QMessageBox.warning(self, "Selection Error", "Please select a profile to capture an image.")

    # Funcție pentru ștergerea unui profil selectat
    def delete_profile(self):
        selected_profile = self.selected_folder.currentItem()
        if selected_profile:
            name = selected_profile.text()
            conn = sqlite3.connect('faces.db')
            c = conn.cursor()
            c.execute("SELECT id FROM profiles WHERE name = ?", (name,))
            profile_id = c.fetchone()[0]
            c.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
            c.execute("DELETE FROM presence WHERE profile_id = ?", (profile_id,))
            c.execute("DELETE FROM descriptors WHERE profile_id = ?", (profile_id,))
            conn.commit()
            conn.close()
            self.update_profile_list()
            update_known_faces()
        else:
            QMessageBox.warning(self, "Selection Error", "Please select a profile to delete.")

    # Funcție pentru actualizarea listei de prezențe
    def update_presence_list(self):
        self.presence_list.clear()
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        c.execute("SELECT profiles.name, presence.timestamp FROM profiles JOIN presence ON profiles.id = presence.profile_id WHERE presence.is_present = 1")
        profiles = c.fetchall()
        for profile in profiles:
            self.presence_list.addItem(f"{profile[0]} - Entered at {profile[1]}")
        conn.close()

    # Funcție pentru căutarea profilurilor în lista de profiluri
    def search_profiles(self, text):
        for i in range(self.selected_folder.count()):
            item = self.selected_folder.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    # Funcție pentru căutarea profilurilor în lista de prezențe
    def search_presence(self, text):
        for i in range(self.presence_list.count()):
            item = self.presence_list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

# Funcția principală a aplicației
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
