
import cv2
import threading
import time
import os
from datetime import datetime
from utils.mediapipe_helper import HandTracker
from utils.frame_pace import FramePacer
from PIL import Image, ImageTk

class VideoRecorder:
    def __init__(self, view, text: str):
        self.view = view
        self.text = text.upper().replace(" ", "  ")
        self.tracker = HandTracker()
        self.is_running = False
        self.out = None
        self.fps = 20
        self.frame_size = (480, 360)

    def start_recording(self):
        if self.is_running:
            return
        self.is_running = True
        self.view.update_status("Recording video...")

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_text = "".join(c if c.isalnum() else "_" for c in self.text.strip())
        filename = f"recordings/{safe_text}_{timestamp}.mp4"
        os.makedirs("recordings", exist_ok=True)

        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(filename, fourcc, self.fps, self.frame_size)

        threading.Thread(target=self._record_loop, daemon=True).start()
        return filename

    def _record_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.view.root.after(0, lambda: self.view.update_status("Camera failed!"))
            self.is_running = False
            return

        char_idx = 0
        pacer = FramePacer(self.fps)
        while self.is_running and char_idx < len(self.text):
            char = self.text[char_idx]
            duration = 1.5 if char != " " else 0.8
            start_time = time.time()

            while time.time() - start_time < duration and self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process + draw
                self.tracker.process_frame(frame)
                frame = self.tracker.draw_landmarks(frame)

                # Resize for video
                resized = cv2.resize(frame, self.frame_size)

                # Write frame
                self.out.write(resized)

                # Show live
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                tk_img = ImageTk.PhotoImage(pil)
                self.view.root.after(0, lambda i=tk_img: self.view.draw_on_canvas(i))

                # Update letter
                self.view.root.after(0, lambda c=char: self.view.update_prediction(c if c != " " else "SPACE"))

                pacer.tick()

            char_idx += 1

        cap.release()
        if self.out:
            self.out.release()
        self.is_running = False
        self.view.root.after(0, lambda: self.view.update_status("Video saved!"))