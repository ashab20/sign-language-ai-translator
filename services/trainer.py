import os
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from model.db_model import GestureDatabase
from utils.mediapipe_helper import LANDMARK_VECTOR_DIM

MODEL_PATH = "data/model.pkl"


def _feature_len(row) -> int | None:
    vec = row[1]
    try:
        return len(vec)
    except TypeError:
        return None


class GestureTrainer:
    def __init__(self):
        self.db = GestureDatabase()

    def train(self):
        rows = self.db.get_all_gestures()
        if not rows:
            return False, "No samples in database."

        target = LANDMARK_VECTOR_DIM
        filtered = [r for r in rows if _feature_len(r) == target]
        skipped = len(rows) - len(filtered)

        def length_histogram() -> str:
            counts: dict[int, int] = {}
            for r in rows:
                n = _feature_len(r)
                if n is not None:
                    counts[n] = counts.get(n, 0) + 1
            parts = [f"{k}→{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])]
            return ", ".join(parts) if parts else "?"

        if not filtered:
            return (
                False,
                f"No samples match current feature size ({target}). "
                f"In DB: {length_histogram()}. Re-record gestures (or clear old rows).",
            )

        if len(filtered) < 10:
            return (
                False,
                f"Need ≥10 samples at {target} features; have {len(filtered)}. "
                f"Skipped {skipped} wrong-size row(s): {length_histogram()}.",
            )

        X = np.array([r[1] for r in filtered], dtype=np.float64)
        y = np.array([r[0] for r in filtered])

        n = len(filtered)
        n_neighbors = min(5, max(3, int(np.sqrt(n))))
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights="distance",
            metric="euclidean",
        )
        model.fit(X, y)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        self.db.close()
        msg = f"Trained on {n} samples → {MODEL_PATH}"
        if skipped:
            msg += f" (skipped {skipped} old/wrong-size rows)"
        return True, msg
