from services.animator import SignAnimator
from services.recorder import GestureRecorder
from services.trainer import GestureTrainer
from services.tts_service import speak_async
from model.ml_model import GestureModel
import tkinter as tk
from tkinter import simpledialog
import os
import cv2
from PIL import Image, ImageTk

class AppController:
    def __init__(self):
        self.ml_model = GestureModel()
        self.trainer = GestureTrainer()

    def train_model(self):
        if hasattr(self, 'view'):
            success, msg = self.trainer.train()
            self.view.update_status(msg)
            if success:
                self.ml_model.predictor._load()

    def use_ai(self):
        self.ml_model.set_view(self.view)
        if self.ml_model.is_running:
            self.view.update_status("Live is already running.")
            return
        self.ml_model.predict_live()

    def stop_ai(self):
        self.ml_model.stop_live()
        self.view.update_status("Stopping AI…")

    def clear_sentence(self):
        if hasattr(self, "view"):
            self.view.clear_sentence_text()
            self.view.update_status("Sentence cleared.")

    def clear_translation(self):
        """Alias for older code paths."""
        self.clear_sentence()

    def speak_sentence(self):
        if not hasattr(self, "view"):
            return
        line = self.view.get_sentence_text()
        if not line:
            self.view.update_status("Nothing to read yet.")
            return
        speak_async(line)
        self.view.update_status("Reading sentence aloud…")

    def speak_translation_line(self):
        """Alias for older code paths."""
        self.speak_sentence()

    def record_gesture(self):
        label = simpledialog.askstring("Gesture Label", "Enter gesture name (e.g., HELLO):")
        if label:
            recorder = GestureRecorder(self.view, label.strip().upper())
            recorder.start()

    def play_text(self, record=False):
        txt = self.view.text_var.get().strip()
        if not txt:
            self.view.update_status("Enter text first!")
            return

        animator = SignAnimator(self.view, txt, record=record)
        animator.start()
        # If you need the path later (e.g. for "Play Last"):
        if record:
            path = animator.get_video_path()
            # store it somewhere or just let the user click "Play Last"

    def play_last_video(self):
        recs = [f for f in os.listdir("recordings") if f.endswith(".mp4")]
        if not recs:
            self.view.update_status("No recordings yet.")
            return
        latest = sorted(recs)[-1]
        path = os.path.join("recordings", latest)

        cap = cv2.VideoCapture(path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (480, 360))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tk_img = ImageTk.PhotoImage(pil)
            self.view.root.after(0, lambda i=tk_img: self.view.draw_on_canvas(i))
        cap.release()
        self.view.update_status(f"Replayed: {latest}")
