import cv2
import threading
from collections import Counter, deque
from PIL import Image, ImageTk

from services.predictor import GesturePredictor, IGNORE_PREDICTIONS
from services.tts_service import speak_async
from utils.mediapipe_helper import HandTracker

# Rolling window for live label stability (frames with a visible hand)
SMOOTH_WINDOW = 15
# Frames without a hand before we treat the sign as "ended"
HAND_GONE_FRAMES = 18


class GestureModel:
    def __init__(self):
        self.tracker = HandTracker()
        self.predictor = GesturePredictor()
        self.is_running = False
        self.view = None

    def set_view(self, view):
        self.view = view

    def stop_live(self):
        self.is_running = False

    def predict_live(self):
        if not self.view or self.is_running:
            return
        self.is_running = True
        self.view.update_status("AI running — show your sign; lower hand when done.")
        threading.Thread(target=self._capture_loop, daemon=True).start()

    @staticmethod
    def _majority(buf: list) -> str:
        if not buf:
            return "—"
        return Counter(buf).most_common(1)[0][0]

    def _capture_loop(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.view.root.after(0, lambda: self.view.update_status("Camera failed!"))
            self.is_running = False
            return

        pred_buffer: deque = deque(maxlen=SMOOTH_WINDOW)
        segment_preds: list = []
        frames_no_hand = 0
        had_hand = False

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            self.tracker.process_frame(frame)
            frame = self.tracker.draw_landmarks(frame)

            landmarks = self.tracker.get_landmarks()
            if landmarks is not None:
                had_hand = True
                frames_no_hand = 0
                pred = self.predictor.predict(landmarks)
                if pred and pred not in IGNORE_PREDICTIONS:
                    pred_buffer.append(pred)
                    segment_preds.append(pred)
                smoothed = self._majority(list(pred_buffer))
                self.view.root.after(0, lambda p=smoothed: self.view.update_prediction(p))
            else:
                frames_no_hand += 1
                if (
                    had_hand
                    and frames_no_hand >= HAND_GONE_FRAMES
                    and segment_preds
                ):
                    final_sign = self._majority(segment_preds)
                    had_hand = False
                    pred_buffer.clear()
                    segment_preds.clear()
                    self.view.root.after(0, lambda f=final_sign: self.view.update_last_completed(f))
                    self.view.root.after(0, lambda p=final_sign: self.view.update_prediction(p))
                    self.view.root.after(0, lambda f=final_sign: self.view.append_translation(f))
                    if (
                        self.view.speak_on_end.get()
                        and final_sign not in IGNORE_PREDICTIONS
                    ):
                        speak_async(final_sign)
                elif not had_hand and frames_no_hand >= HAND_GONE_FRAMES:
                    pred_buffer.clear()
                    segment_preds.clear()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb).resize((480, 360))
            tk_img = ImageTk.PhotoImage(pil_img)
            self.view.root.after(0, lambda img=tk_img: self.view.draw_on_canvas(img))

        cap.release()
        self.is_running = False
        self.view.root.after(0, lambda: self.view.update_status("AI stopped."))
