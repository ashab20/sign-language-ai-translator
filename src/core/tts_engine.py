"""Text-to-speech engine using pyttsx3 (offline, cross-platform)."""

import threading


class TTSEngine:
    """Speaks text asynchronously so it doesn't block the UI."""

    def __init__(self):
        self._lock = threading.Lock()
        self._speaking = False

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def speak(self, text: str):
        if not text or not text.strip():
            return
        if self._speaking:
            return

        thread = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
        thread.start()

    def _speak_thread(self, text: str):
        self._speaking = True
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.setProperty("volume", 0.9)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass
        finally:
            self._speaking = False
