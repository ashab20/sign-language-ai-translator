"""About tab: university info, project description, and help."""

import tkinter as tk

import ttkbootstrap as tbs
from PIL import Image, ImageTk
from ttkbootstrap.constants import *

from config import (
    PROJECT_SUBTITLE,
    PROJECT_TITLE,
    STUDENT_NAME,
    SUPERVISOR_NAME,
    UNIVERSITY_NAME,
)
from src.utils.university_logo import resolve_university_logo_png


class AboutTab(tbs.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self._logo_photo = None
        self._build_ui()

    def _build_ui(self):
        container = tbs.Frame(self)
        container.pack(fill=BOTH, expand=True, padx=40, pady=20)

        # University logo
        logo_frame = tbs.Frame(container)
        logo_frame.pack(pady=(0, 16))

        self._load_logo(logo_frame)

        # University name
        tbs.Label(
            container, text=UNIVERSITY_NAME,
            font=("Segoe UI", 20, "bold"), bootstyle="primary",
            anchor=CENTER,
        ).pack(fill=X, pady=(0, 8))

        # Separator
        tbs.Separator(container, bootstyle="primary").pack(fill=X, pady=8)

        # Project title
        tbs.Label(
            container, text=PROJECT_TITLE,
            font=("Segoe UI", 18, "bold"), anchor=CENTER,
        ).pack(fill=X, pady=(8, 4))

        tbs.Label(
            container, text=PROJECT_SUBTITLE,
            font=("Segoe UI", 13), bootstyle="secondary", anchor=CENTER,
        ).pack(fill=X, pady=(0, 16))

        # Project info cards
        info_frame = tbs.Frame(container)
        info_frame.pack(fill=X, pady=8)

        info_frame.columnconfigure(0, weight=1)
        info_frame.columnconfigure(1, weight=1)

        # Student card
        student_card = tbs.Labelframe(info_frame, text="Student", bootstyle="info", padding=15)
        student_card.grid(row=0, column=0, sticky=NSEW, padx=(0, 8))

        tbs.Label(
            student_card, text=STUDENT_NAME,
            font=("Segoe UI", 14, "bold"), anchor=CENTER,
        ).pack(fill=X)

        # Supervisor card
        super_card = tbs.Labelframe(info_frame, text="Supervisor", bootstyle="success", padding=15)
        super_card.grid(row=0, column=1, sticky=NSEW, padx=(8, 0))

        tbs.Label(
            super_card, text=SUPERVISOR_NAME,
            font=("Segoe UI", 14, "bold"), anchor=CENTER,
        ).pack(fill=X)

        # Separator
        tbs.Separator(container, bootstyle="secondary").pack(fill=X, pady=16)

        # Description
        desc_frame = tbs.Labelframe(
            container, text="About This Project", bootstyle="secondary", padding=15,
        )
        desc_frame.pack(fill=X, pady=(0, 16))

        description = (
            "This application uses artificial intelligence to translate sign language "
            "gestures into text and speech. It uses MediaPipe for real-time hand tracking "
            "and machine learning (Random Forest / MLP) for gesture classification.\n\n"
            "The system can recognize trained ASL (American Sign Language) signs, build "
            "them into sentences, and convert the text to speech -- helping people who "
            "communicate through sign language to be understood by everyone."
        )

        tbs.Label(
            desc_frame, text=description,
            font=("Segoe UI", 11), wraplength=700,
            justify=LEFT,
        ).pack(fill=X)

        # Quick start guide
        help_frame = tbs.Labelframe(
            container, text="Quick Start Guide", bootstyle="warning", padding=15,
        )
        help_frame.pack(fill=X)

        steps = [
            "1.  Go to the Train tab and start the camera",
            "2.  Enter a sign name (e.g., 'A', 'HELLO') and click Record",
            "3.  Hold the sign steady while 30 frames are captured",
            "4.  Repeat for all signs you want to detect (min. 20 samples each)",
            "5.  Click 'Train Model' to train the AI classifier",
            "6.  Go to the Detect tab to start recognizing signs in real-time",
            "7.  Signs build into sentences -- click 'Speak' to hear them!",
        ]

        for step in steps:
            tbs.Label(
                help_frame, text=step,
                font=("Segoe UI", 11), anchor=W,
            ).pack(fill=X, pady=2)

        # Version
        tbs.Label(
            container, text="Version 2.0  |  Built with Python, MediaPipe & scikit-learn",
            font=("Segoe UI", 9), bootstyle="secondary", anchor=CENTER,
        ).pack(fill=X, pady=(16, 0))

    def _load_logo(self, parent):
        logo_path = resolve_university_logo_png()
        if logo_path:
            try:
                img = Image.open(logo_path)
                img = img.resize((120, 120), Image.Resampling.LANCZOS)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                self._logo_photo = ImageTk.PhotoImage(img)
                tbs.Label(parent, image=self._logo_photo).pack()
                return
            except Exception:
                pass

        # Placeholder when no logo is available
        canvas = tk.Canvas(parent, width=120, height=120, bg="#e9ecef", highlightthickness=0)
        canvas.pack()
        canvas.create_oval(10, 10, 110, 110, outline="#6c757d", width=2, dash=(4, 4))
        canvas.create_text(60, 50, text="JU", font=("Segoe UI", 14, "bold"), fill="#6c757d")
        canvas.create_text(
            60, 75,
            text="Add assets/ju.svg\nor ju.png",
            font=("Segoe UI", 8), fill="#adb5bd",
        )
