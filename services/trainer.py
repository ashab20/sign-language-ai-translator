import os
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from model.db_model import GestureDatabase

MODEL_PATH = "data/model.pkl"


class GestureTrainer:
    def __init__(self):
        self.db = GestureDatabase()

    def train(self):
        rows = self.db.get_all_gestures()
        if len(rows) < 10:
            return False, "Need ≥10 samples to train."

        X = np.array([r[1] for r in rows])
        y = np.array([r[0] for r in rows])

        n_neighbors = min(5, max(3, int(np.sqrt(len(rows)))))
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights="distance",
            metric="euclidean",
        )
        model.fit(X, y)

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        self.db.close()
        return True, f"Trained on {len(rows)} samples → {MODEL_PATH}"
