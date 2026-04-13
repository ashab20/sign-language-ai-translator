import os
import cv2
import threading
import time
from datetime import datetime
from PIL import Image, ImageTk
from utils.mediapipe_helper import HandTracker
from utils.frame_pace import FramePacer
from services.predictor import GesturePredictor


class SignAnimator:
    """
    Plays a sentence as a sequence of signs.
    Optional: record the whole animation to recordings/<TEXT>_<timestamp>.mp4
    """

    def __init__(self, view, text: str, record: bool = False):
        self.view = view
        self.text = text.upper().replace(" ", "  ")   # double-space = pause
        self.record = record
        self.tracker = HandTracker()
        self.predictor = GesturePredictor()
        self.is_running = False
        self.current_idx = 0
        self.play_fps = 20.0

        # Video writer (only if record=True)
        self.out = None
        self.video_path = None
        if self.record:
            self._prepare_video_writer()

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.current_idx = 0
        self.view.update_prediction("")
        self.view.update_status(f"Playing: {self.text.replace('  ', ' ')}")
        threading.Thread(target=self._play_loop, daemon=True).start()

    def _prepare_video_writer(self):
        safe = "".join(c if c.isalnum() else "_" for c in self.text.strip())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = f"recordings/{safe}_{timestamp}.mp4"
        os.makedirs("recordings", exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.video_path, fourcc, self.play_fps, (480, 360))

    def _play_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.view.root.after(0, lambda: self.view.update_status("Camera failed!"))
            self.is_running = False
            return

        pacer = FramePacer(self.play_fps)
        while self.is_running and self.current_idx < len(self.text):
            char = self.text[self.current_idx]

            # ----- show current letter -----
            display_char = char if char != " " else "SPACE"
            self.view.root.after(0, lambda c=display_char: self.view.update_prediction(c))

            # ----- timing -----
            duration = 1.5 if char != " " else 0.8
            start = time.time()

            while (time.time() - start) < duration and self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                # MediaPipe processing
                self.tracker.process_frame(frame)
                frame = self.tracker.draw_landmarks(frame)

                # ----- prediction (shows the *trained* sign name) -----
                lm = self.tracker.get_landmarks()
                pred = self.predictor.predict(lm) if lm is not None else "NO TRACK"
                self.view.root.after(0, lambda p=pred: self.view.update_prediction(p))

                # ----- live preview -----
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb).resize((480, 360))
                tk_img = ImageTk.PhotoImage(pil)
                self.view.root.after(0, lambda i=tk_img: self.view.draw_on_canvas(i))

                # ----- record frame (if requested) -----
                if self.record and self.out:
                    small = cv2.resize(frame, (480, 360))
                    self.out.write(small)

                pacer.tick()

            self.current_idx += 1

        cap.release()
        if self.out:
            self.out.release()

        self.is_running = False
        msg = f"Finished! Video saved: {os.path.basename(self.video_path)}" if self.record else "Animation finished!"
        self.view.root.after(0, lambda: self.view.update_status(msg))

    def get_video_path(self):
        return self.video_path