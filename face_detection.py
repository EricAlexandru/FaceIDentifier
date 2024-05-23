import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")
        
        # Initialize camera
        self.video_capture = cv2.VideoCapture(0)
        
        # Create GUI elements
        self.create_widgets()
        
        # Start video streaming
        self.show_video_feed()
    
    def create_widgets(self):
        # Create a label to display video feed
        self.video_label = tk.Label(self.root)
        self.video_label.pack()
        
        # Create a button to capture photo
        self.capture_button = ttk.Button(self.root, text="Capture Photo", command=self.capture_photo)
        self.capture_button.pack()
    
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
            cv2.imwrite("captured_photo.jpg", frame)
            print("Photo captured!")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
