import threading

from services.predictor import IGNORE_PREDICTIONS


def speak_async(text: str) -> None:
    """Speak text on a background thread so the GUI stays responsive."""

    def _run():
        t = (text or "").strip()
        u = t.upper()
        if not t or u in IGNORE_PREDICTIONS or u in ("-",):
            return
        try:
            import pyttsx3

            engine = pyttsx3.init()
            try:
                engine.setProperty("rate", 175)
            except Exception:
                pass
            engine.say(t)
            engine.runAndWait()
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()
