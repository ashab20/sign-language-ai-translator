"""Main application window with tabbed interface."""

import ttkbootstrap as tbs
from ttkbootstrap.constants import *

from config import APP_TITLE, WINDOW_SIZE
from src.core.classifier import SignClassifier
from src.core.hand_detector import HandDetector
from src.core.sentence_builder import SentenceBuilder
from src.core.tts_engine import TTSEngine
from src.ui.about_tab import AboutTab
from src.ui.detect_tab import DetectTab
from src.ui.train_tab import TrainTab


class SignLanguageApp(tbs.Window):
    def __init__(self):
        super().__init__(
            title=APP_TITLE,
            themename="cosmo",
            size=(int(WINDOW_SIZE.split("x")[0]), int(WINDOW_SIZE.split("x")[1])),
            resizable=(True, True),
        )

        self.detector = HandDetector()
        self.classifier = SignClassifier()
        self.sentence_builder = SentenceBuilder()
        self.tts = TTSEngine()

        self.classifier.load_model()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        self.notebook = tbs.Notebook(self, bootstyle="primary")
        self.notebook.pack(fill=BOTH, expand=True, padx=8, pady=8)

        self.detect_tab = DetectTab(self.notebook, self)
        self.train_tab = TrainTab(self.notebook, self)
        self.about_tab = AboutTab(self.notebook, self)

        self.notebook.add(self.detect_tab, text="  Detect Signs  ")
        self.notebook.add(self.train_tab, text="  Train Model  ")
        self.notebook.add(self.about_tab, text="  About  ")

        self.status_var = tbs.StringVar(value="Ready")
        status_bar = tbs.Label(
            self, textvariable=self.status_var,
            bootstyle="inverse-secondary",
            font=("Segoe UI", 10),
            padding=(10, 4),
        )
        status_bar.pack(fill=X, side=BOTTOM)

    def set_status(self, message: str):
        self.status_var.set(message)

    def reload_model(self):
        self.classifier.load_model()

    def _on_close(self):
        self.detect_tab.stop_camera()
        self.train_tab.stop_camera()
        self.detector.release()
        self.destroy()
