"""Train tab: record signs, manage data, train model, view accuracy reports."""

import threading
import tkinter as tk

import cv2
import numpy as np
import ttkbootstrap as tbs
from PIL import Image, ImageTk
from ttkbootstrap.constants import *

from config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    MIN_SAMPLES_PER_SIGN,
    RECORDING_FRAMES,
    TOTAL_FEATURES,
)
from src.data import database as db


class TrainTab(tbs.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._cap = None
        self._running = False
        self._recording = False
        self._recorded_count = 0
        self._photo = None
        self._after_id = None

        self._build_ui()
        self._refresh_data_table()

    def _build_ui(self):
        # Main horizontal layout
        main = tbs.Frame(self)
        main.pack(fill=BOTH, expand=True, padx=8, pady=8)

        # LEFT: camera + recording controls
        left = tbs.Frame(main)
        left.pack(side=LEFT, fill=BOTH, expand=True)

        cam_frame = tbs.Labelframe(left, text="Camera Preview", bootstyle="primary", padding=5)
        cam_frame.pack(fill=BOTH, expand=True)

        self.canvas = tk.Canvas(cam_frame, width=480, height=360, bg="#1a1a2e")
        self.canvas.pack(padx=4, pady=4)

        # Camera controls
        cam_btns = tbs.Frame(cam_frame)
        cam_btns.pack(fill=X, pady=4)

        self.btn_cam_start = tbs.Button(
            cam_btns, text="Start Camera", bootstyle="success",
            command=self.start_camera, width=14,
        )
        self.btn_cam_start.pack(side=LEFT, padx=4)

        self.btn_cam_stop = tbs.Button(
            cam_btns, text="Stop Camera", bootstyle="danger",
            command=self.stop_camera, width=14, state=DISABLED,
        )
        self.btn_cam_stop.pack(side=LEFT, padx=4)

        # Recording controls
        rec_frame = tbs.Labelframe(left, text="Record New Sign", bootstyle="info", padding=10)
        rec_frame.pack(fill=X, pady=(8, 0))

        input_row = tbs.Frame(rec_frame)
        input_row.pack(fill=X, pady=(0, 8))

        tbs.Label(input_row, text="Sign Name:", font=("Segoe UI", 11)).pack(side=LEFT, padx=(0, 8))

        self.sign_entry = tbs.Entry(input_row, font=("Segoe UI", 12), width=16)
        self.sign_entry.pack(side=LEFT, padx=(0, 8))

        self.btn_record = tbs.Button(
            input_row, text=f"Record ({RECORDING_FRAMES} frames)",
            bootstyle="warning", command=self._start_recording, width=20,
        )
        self.btn_record.pack(side=LEFT, padx=4)

        self.record_progress = tbs.Progressbar(
            rec_frame, bootstyle="warning-striped",
            maximum=RECORDING_FRAMES, length=400,
        )
        self.record_progress.pack(fill=X, pady=4)

        self.record_status = tbs.Label(
            rec_frame, text="Enter a sign name and click Record",
            font=("Segoe UI", 10), bootstyle="secondary",
        )
        self.record_status.pack(fill=X)

        # RIGHT: data management + training
        right = tbs.Frame(main, width=350)
        right.pack(side=RIGHT, fill=Y, padx=(8, 0))
        right.pack_propagate(False)

        # Data table
        data_frame = tbs.Labelframe(right, text="Recorded Signs", bootstyle="success", padding=5)
        data_frame.pack(fill=BOTH, expand=True)

        # Treeview for sign data
        tree_frame = tbs.Frame(data_frame)
        tree_frame.pack(fill=BOTH, expand=True)

        self.tree = tbs.Treeview(
            tree_frame, columns=("sign", "count", "status"),
            show="headings", height=10, bootstyle="info",
        )
        self.tree.heading("sign", text="Sign")
        self.tree.heading("count", text="Samples")
        self.tree.heading("status", text="Status")
        self.tree.column("sign", width=100, anchor=CENTER)
        self.tree.column("count", width=80, anchor=CENTER)
        self.tree.column("status", width=100, anchor=CENTER)

        scrollbar = tbs.Scrollbar(tree_frame, orient=VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        # Data buttons
        data_btns = tbs.Frame(data_frame)
        data_btns.pack(fill=X, pady=(4, 0))

        self.btn_delete = tbs.Button(
            data_btns, text="Delete Selected", bootstyle="danger-outline",
            command=self._delete_selected, width=16,
        )
        self.btn_delete.pack(side=LEFT, padx=4)

        self.btn_refresh = tbs.Button(
            data_btns, text="Refresh", bootstyle="secondary",
            command=self._refresh_data_table, width=10,
        )
        self.btn_refresh.pack(side=RIGHT, padx=4)

        # Training section
        train_frame = tbs.Labelframe(right, text="Train Model", bootstyle="warning", padding=10)
        train_frame.pack(fill=X, pady=(8, 0))

        # Model type selector
        type_row = tbs.Frame(train_frame)
        type_row.pack(fill=X, pady=(0, 8))

        tbs.Label(type_row, text="Model:", font=("Segoe UI", 10)).pack(side=LEFT)

        self.model_type_var = tbs.StringVar(value="random_forest")
        tbs.Radiobutton(
            type_row, text="Random Forest", variable=self.model_type_var,
            value="random_forest", bootstyle="success-toolbutton",
        ).pack(side=LEFT, padx=4)
        tbs.Radiobutton(
            type_row, text="MLP Neural Net", variable=self.model_type_var,
            value="mlp", bootstyle="info-toolbutton",
        ).pack(side=LEFT, padx=4)

        self.btn_train = tbs.Button(
            train_frame, text="Train Model", bootstyle="success",
            command=self._start_training, width=30,
        )
        self.btn_train.pack(fill=X, pady=4)

        self.train_progress = tbs.Progressbar(
            train_frame, bootstyle="success-striped", maximum=100,
        )
        self.train_progress.pack(fill=X, pady=4)

        self.train_status = tbs.Label(
            train_frame, text="", font=("Segoe UI", 9),
            bootstyle="secondary", wraplength=300,
        )
        self.train_status.pack(fill=X)

        # Results section
        results_frame = tbs.Labelframe(right, text="Training Results", bootstyle="info", padding=5)
        results_frame.pack(fill=X, pady=(8, 0))

        self.results_text = tk.Text(
            results_frame, height=6, font=("Consolas", 9),
            wrap=tk.WORD, state=DISABLED, bg="#f8f9fa",
            relief="flat", padx=6, pady=4,
        )
        self.results_text.pack(fill=X)

    def start_camera(self):
        if self._running:
            return

        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            self.app.set_status("Error: Could not open camera")
            return

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self._running = True
        self.btn_cam_start.configure(state=DISABLED)
        self.btn_cam_stop.configure(state=NORMAL)
        self.app.set_status("Training camera active")
        self._update_frame()

    def stop_camera(self):
        self._running = False
        self._recording = False
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self._cap:
            self._cap.release()
            self._cap = None

        self.btn_cam_start.configure(state=NORMAL)
        self.btn_cam_stop.configure(state=DISABLED)

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

        if self._recording:
            features = self.app.detector.get_features()
            if features is not None:
                self._recorded_frames.append(features.tolist())
                self._recorded_count += 1
                self.record_progress["value"] = self._recorded_count
                self.record_status.configure(
                    text=f"Recording: {self._recorded_count}/{RECORDING_FRAMES}",
                    bootstyle="warning",
                )

                # Draw recording indicator
                cv2.putText(
                    frame, f"REC {self._recorded_count}/{RECORDING_FRAMES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )
                cv2.circle(frame, (470, 20), 8, (0, 0, 255), -1)

            if self._recorded_count >= RECORDING_FRAMES:
                self._finish_recording()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((480, 360), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(240, 180, image=self._photo, anchor=CENTER)

        self._after_id = self.after(33, self._update_frame)

    def _start_recording(self):
        sign_name = self.sign_entry.get().strip()
        if not sign_name:
            self.record_status.configure(
                text="Please enter a sign name first!",
                bootstyle="danger",
            )
            return

        if not self._running:
            self.record_status.configure(
                text="Start the camera first!",
                bootstyle="danger",
            )
            return

        self._recording = True
        self._recorded_count = 0
        self._recorded_frames = []
        self.record_progress["value"] = 0
        self.btn_record.configure(state=DISABLED)
        self.record_status.configure(
            text="Recording... Hold your sign steady!",
            bootstyle="warning",
        )
        self.app.set_status(f"Recording sign: {sign_name.upper()}")

    def _finish_recording(self):
        self._recording = False
        self.btn_record.configure(state=NORMAL)

        sign_name = self.sign_entry.get().strip().upper()
        count = len(self._recorded_frames)

        if count == 0:
            self.record_status.configure(
                text="No frames captured. Make sure your hand is visible!",
                bootstyle="danger",
            )
            return

        db.add_samples_bulk(sign_name, self._recorded_frames)
        self.record_status.configure(
            text=f"Saved {count} samples for '{sign_name}'",
            bootstyle="success",
        )
        self.app.set_status(f"Recorded {count} samples for '{sign_name}'")
        self._refresh_data_table()
        self._recorded_frames = []

    def _refresh_data_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        counts = db.get_label_counts()
        for label, count in counts.items():
            status = "Ready" if count >= MIN_SAMPLES_PER_SIGN else f"Need {MIN_SAMPLES_PER_SIGN - count} more"
            tag = "ready" if count >= MIN_SAMPLES_PER_SIGN else "need_more"
            self.tree.insert("", END, values=(label, count, status), tags=(tag,))

        self.tree.tag_configure("ready", foreground="#28a745")
        self.tree.tag_configure("need_more", foreground="#dc3545")

    def _delete_selected(self):
        selected = self.tree.selection()
        if not selected:
            return

        for item in selected:
            values = self.tree.item(item, "values")
            label = values[0]
            db.delete_label(label)

        self._refresh_data_table()
        self.app.set_status("Selected signs deleted")

    def _start_training(self):
        counts = db.get_label_counts()
        if not counts:
            self.train_status.configure(text="No data! Record some signs first.", bootstyle="danger")
            return

        insufficient = [l for l, c in counts.items() if c < MIN_SAMPLES_PER_SIGN]
        if insufficient:
            self.train_status.configure(
                text=f"Not enough samples for: {', '.join(insufficient)}. "
                     f"Need at least {MIN_SAMPLES_PER_SIGN} per sign.",
                bootstyle="danger",
            )
            return

        self.btn_train.configure(state=DISABLED)
        self.train_status.configure(text="Training started...", bootstyle="info")
        self.train_progress["value"] = 0

        model_type = self.model_type_var.get()

        thread = threading.Thread(target=self._train_thread, args=(model_type,), daemon=True)
        thread.start()

    def _train_thread(self, model_type: str):
        samples = db.get_all_samples()
        if not samples:
            self.after(0, self._training_failed, "No samples found.")
            return

        X = np.array([s[1] for s in samples if len(s[1]) == TOTAL_FEATURES], dtype=np.float64)
        y = np.array([s[0] for s in samples if len(s[1]) == TOTAL_FEATURES])

        if len(X) == 0:
            self.after(0, self._training_failed, "No valid samples (feature size mismatch).")
            return

        def progress_cb(pct, msg):
            self.after(0, self._update_train_progress, pct, msg)

        result = self.app.classifier.train(X, y, model_type=model_type, progress_callback=progress_cb)
        self.after(0, self._training_complete, result)

    def _update_train_progress(self, pct: int, msg: str):
        self.train_progress["value"] = pct
        self.train_status.configure(text=msg, bootstyle="info")

    def _training_complete(self, result: dict):
        self.btn_train.configure(state=NORMAL)

        if not result["success"]:
            self._training_failed("Training failed. Check your data.")
            return

        self.train_status.configure(
            text=f"Training complete! Accuracy: {result['accuracy']:.1%}",
            bootstyle="success",
        )

        report_lines = [
            f"Model: {result['model_type'].replace('_', ' ').title()}",
            f"Samples: {result['n_samples']} | Classes: {result['n_classes']}",
            f"Test Accuracy: {result['accuracy']:.1%}",
            f"CV Accuracy: {result['cv_accuracy']:.1%} (+/- {result['cv_std']:.1%})",
            "",
            result["classification_report"],
        ]

        self.results_text.configure(state=NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", "\n".join(report_lines))
        self.results_text.configure(state=DISABLED)

        self.app.reload_model()
        self.app.set_status(
            f"Model trained: {result['accuracy']:.1%} accuracy on {result['n_classes']} signs"
        )

    def _training_failed(self, message: str):
        self.btn_train.configure(state=NORMAL)
        self.train_status.configure(text=message, bootstyle="danger")
