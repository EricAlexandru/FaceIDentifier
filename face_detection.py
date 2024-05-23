import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")
        
        # URL for the video stream
        self.video_url = "http://192.168.0.234:4747/video"
        
        # Initialize video capture
        self.video_capture = cv2.VideoCapture(self.video_url)
        
        # Create GUI elements
        self.create_widgets()
        
        # Profiles
        self.profiles = {}
        self.selected_profile = None
    
    def create_widgets(self):
        # Create a label to display video feed
        self.video_label = tk.Label(self.root)
        self.video_label.pack()
        
        # Create a button to capture photo
        self.capture_button = ttk.Button(self.root, text="Capture Photo", command=self.capture_photo)
        self.capture_button.pack()
        
        # Create a combobox to select profiles
        self.profile_combobox = ttk.Combobox(self.root, values=["Create New Profile"])
        self.profile_combobox.pack()
        self.profile_combobox.bind("<<ComboboxSelected>>", self.select_profile)
        
    def show_video_feed(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            self.video_label.img = img
            self.video_label.configure(image=img)
        
        self.video_label.after(10, self.show_video_feed)
    
    def capture_photo(self):
        ret, frame = self.video_capture.read()
        if ret:
            if self.selected_profile:
                profile_dir = os.path.join("profiles", self.selected_profile)
                if not os.path.exists(profile_dir):
                    os.makedirs(profile_dir)
                photo_path = os.path.join(profile_dir, f"photo_{len(self.profiles[self.selected_profile])}.jpg")
                cv2.imwrite(photo_path, frame)
                self.profiles[self.selected_profile].append(photo_path)
                print(f"Photo captured for profile '{self.selected_profile}'!")
            else:
                print("Please select a profile first.")
    
    def select_profile(self, event):
        selected_profile = self.profile_combobox.get()
        if selected_profile == "Create New Profile":
            new_profile = tk.simpledialog.askstring("New Profile", "Enter Profile Name:")
            if new_profile:
                self.profiles[new_profile] = []
                self.profile_combobox["values"] = list(self.profiles.keys()) + ["Create New Profile"]
                self.profile_combobox.set(new_profile)
                self.selected_profile = new_profile
        else:
            self.selected_profile = selected_profile

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    app.show_video_feed()  # Start showing video feed
    root.mainloop()
