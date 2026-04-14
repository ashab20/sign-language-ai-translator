"""Entry point for the Sign Language AI Translator application."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ui.app_window import SignLanguageApp


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    app = SignLanguageApp()
    app.mainloop()


if __name__ == "__main__":
    main()
