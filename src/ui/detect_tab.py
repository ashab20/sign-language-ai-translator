"""Detect tab: live camera feed with sign detection, sentence builder, and TTS."""

import tkinter as tk

import cv2
import numpy as np
import ttkbootstrap as tbs
from PIL import Image, ImageTk
from ttkbootstrap.constants import *

from config import CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH


class DetectTab(tbs.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._cap = None
        self._running = False
        self._photo = None
        self._after_id = None

        self._build_ui()

    def _build_ui(self):
        # Top section: camera + info panel
        top = tbs.Frame(self)
        top.pack(fill=BOTH, expand=True, padx=8, pady=(8, 4))

        # Camera frame (left)
        cam_frame = tbs.Labelframe(top, text="Camera", bootstyle="primary", padding=5)
        cam_frame.pack(side=LEFT, fill=BOTH, expand=True)

        self.canvas = tk.Canvas(cam_frame, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, bg="#1a1a2e")
        self.canvas.pack(padx=4, pady=4)

        cam_controls = tbs.Frame(cam_frame)
        cam_controls.pack(fill=X, pady=(4, 0))

        self.btn_start = tbs.Button(
            cam_controls, text="Start Camera", bootstyle="success",
            command=self.start_camera, width=16,
        )
        self.btn_start.pack(side=LEFT, padx=4)

        self.btn_stop = tbs.Button(
            cam_controls, text="Stop Camera", bootstyle="danger",
            command=self.stop_camera, width=16, state=DISABLED,
        )
        self.btn_stop.pack(side=LEFT, padx=4)

        # Info panel (right)
        info_frame = tbs.Frame(top, width=300)
        info_frame.pack(side=RIGHT, fill=Y, padx=(8, 0))
        info_frame.pack_propagate(False)

        # Detected sign display
        sign_frame = tbs.Labelframe(info_frame, text="Detected Sign", bootstyle="info", padding=10)
        sign_frame.pack(fill=X, pady=(0, 8))

        self.sign_label = tbs.Label(
            sign_frame, text="--", font=("Segoe UI", 48, "bold"),
            bootstyle="primary", anchor=CENTER,
        )
        self.sign_label.pack(fill=X, pady=8)

        # Confidence bar
        conf_frame = tbs.Frame(sign_frame)
        conf_frame.pack(fill=X)
        tbs.Label(conf_frame, text="Confidence:", font=("Segoe UI", 10)).pack(side=LEFT)
        self.conf_label = tbs.Label(conf_frame, text="0%", font=("Segoe UI", 10, "bold"))
        self.conf_label.pack(side=RIGHT)

        self.conf_bar = tbs.Progressbar(sign_frame, bootstyle="success-striped", length=250, maximum=100)
        self.conf_bar.pack(fill=X, pady=(4, 0))

        # Status indicator
        self.status_label = tbs.Label(
            info_frame, text="Camera stopped",
            font=("Segoe UI", 11), bootstyle="secondary", anchor=CENTER,
        )
        self.status_label.pack(fill=X, pady=8)

        # Model status
        model_text = "Model loaded" if self.app.classifier.is_loaded else "No model - train first!"
        model_style = "success" if self.app.classifier.is_loaded else "warning"
        self.model_label = tbs.Label(
            info_frame, text=model_text,
            font=("Segoe UI", 10), bootstyle=model_style, anchor=CENTER,
        )
        self.model_label.pack(fill=X, pady=(0, 8))

        # Bottom section: sentence builder
        bottom = tbs.Labelframe(self, text="Sentence Builder", bootstyle="success", padding=10)
        bottom.pack(fill=X, padx=8, pady=(4, 8))

        self.sentence_text = tk.Text(
            bottom, height=3, font=("Segoe UI", 16),
            wrap=tk.WORD, state=DISABLED, bg="#f0f8ff",
            relief="flat", padx=10, pady=8,
        )
        self.sentence_text.pack(fill=X, pady=(0, 8))

        btn_row = tbs.Frame(bottom)
        btn_row.pack(fill=X)

        self.btn_speak = tbs.Button(
            btn_row, text="Speak Sentence", bootstyle="info",
            command=self._speak, width=18,
        )
        self.btn_speak.pack(side=LEFT, padx=4)

        self.btn_backspace = tbs.Button(
            btn_row, text="Backspace", bootstyle="warning",
            command=self._backspace, width=14,
        )
        self.btn_backspace.pack(side=LEFT, padx=4)

        self.btn_clear = tbs.Button(
            btn_row, text="Clear Sentence", bootstyle="danger-outline",
            command=self._clear_sentence, width=16,
        )
        self.btn_clear.pack(side=LEFT, padx=4)

        tbs.Label(
            btn_row,
            text="Tip: Hold a sign steady to add it to the sentence",
            font=("Segoe UI", 9), bootstyle="secondary",
        ).pack(side=RIGHT, padx=8)

    def start_camera(self):
        if self._running:
            return

        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            self.app.set_status("Error: Could not open camera")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        self._running = True
        self.btn_start.configure(state=DISABLED)
        self.btn_stop.configure(state=NORMAL)
        self.status_label.configure(text="Detecting...", bootstyle="success")
        self.app.set_status("Camera active - show signs to detect")

        self._update_model_label()
        self._update_frame()

    def stop_camera(self):
        self._running = False
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self._cap:
            self._cap.release()
            self._cap = None

        self.btn_start.configure(state=NORMAL)
        self.btn_stop.configure(state=DISABLED)
        self.status_label.configure(text="Camera stopped", bootstyle="secondary")
        self.app.set_status("Ready")

    def _update_frame(self):
        if not self._running or not self._cap:
            return

        ret, frame = self._cap.read()
        if not ret:
            self._after_id = self.after(33, self._update_frame)
            return

        frame = cv2.flip(frame, 1)
        self.app.detector.process_frame(frame)
        frame = self.app.detector.draw_landmarks(frame)

        features = self.app.detector.get_features()
        has_hands = self.app.detector.has_hands()

        label, confidence = "", 0.0
        if features is not None and self.app.classifier.is_loaded:
            label, confidence = self.app.classifier.predict(features)

        smoothed = self.app.sentence_builder.update(label, has_hands)

        self._update_sign_display(smoothed, confidence, has_hands)
        self._update_sentence_display()

        if self.app.sentence_builder.sign_just_confirmed:
            self.app.set_status(f"Sign confirmed: {self.app.sentence_builder._last_confirmed}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(
            CAMERA_WIDTH // 2, CAMERA_HEIGHT // 2,
            image=self._photo, anchor=CENTER,
        )

        self._after_id = self.after(33, self._update_frame)

    def _update_sign_display(self, label: str, confidence: float, has_hands: bool):
        if not has_hands:
            self.sign_label.configure(text="--")
            self.conf_bar["value"] = 0
            self.conf_label.configure(text="0%")
            return

        display = label if label else "..."
        self.sign_label.configure(text=display)

        pct = int(confidence * 100)
        self.conf_bar["value"] = pct
        self.conf_label.configure(text=f"{pct}%")

        if pct >= 70:
            self.conf_bar.configure(bootstyle="success-striped")
        elif pct >= 40:
            self.conf_bar.configure(bootstyle="warning-striped")
        else:
            self.conf_bar.configure(bootstyle="danger-striped")

    def _update_sentence_display(self):
        sentence = self.app.sentence_builder.sentence
        self.sentence_text.configure(state=NORMAL)
        self.sentence_text.delete("1.0", tk.END)
        self.sentence_text.insert("1.0", sentence if sentence else "Signs will appear here...")
        self.sentence_text.configure(state=DISABLED)

    def _update_model_label(self):
        if self.app.classifier.is_loaded:
            self.model_label.configure(text="Model loaded", bootstyle="success")
        else:
            self.model_label.configure(text="No model - train first!", bootstyle="warning")

    def _speak(self):
        sentence = self.app.sentence_builder.sentence
        if sentence:
            self.app.tts.speak(sentence)
            self.app.set_status(f"Speaking: {sentence}")

    def _backspace(self):
        self.app.sentence_builder.backspace()
        self._update_sentence_display()

    def _clear_sentence(self):
        self.app.sentence_builder.clear()
        self._update_sentence_display()
        self.app.set_status("Sentence cleared")
