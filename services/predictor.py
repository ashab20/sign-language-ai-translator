import os
from typing import Optional

import joblib
import numpy as np

MODEL_PATH = "data/model.pkl"
MIN_CONFIDENCE = 0.22

# Not real class labels — do not add to smoothing buffer or TTS
IGNORE_PREDICTIONS = frozenset({"NO MODEL", "—", "RETRAIN MODEL"})


class GesturePredictor:
    def __init__(self):
        self.model = None
        self._load()

    def _load(self):
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)

    def _expected_n_features(self) -> Optional[int]:
        if self.model is None:
            return None
        n = getattr(self.model, "n_features_in_", None)
        if n is not None:
            return int(n)
        return None

    def predict(self, landmarks):
        if self.model is None or landmarks is None:
            return "NO MODEL"
        arr = np.array(landmarks, dtype=np.float64).reshape(1, -1)
        got = arr.shape[1]
        expected = self._expected_n_features()
        if expected is not None and got != expected:
            return "RETRAIN MODEL"

        try:
            proba = self.model.predict_proba(arr)[0]
            best_i = int(np.argmax(proba))
            if float(proba[best_i]) < MIN_CONFIDENCE:
                return "—"
            label = self.model.classes_[best_i]
            return str(label).upper()
        except ValueError:
            # e.g. sklearn feature count mismatch on older pickles
            return "RETRAIN MODEL"
        except Exception:
            try:
                return str(self.model.predict(arr)[0]).upper()
            except ValueError:
                return "RETRAIN MODEL"
