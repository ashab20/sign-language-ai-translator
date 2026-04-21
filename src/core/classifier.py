"""ML classifier with Random Forest and MLP, including evaluation pipeline."""

import os
import warnings

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier

from config import (
    CONFUSION_MATRIX_PATH,
    CV_FOLDS,
    MIN_CONFIDENCE,
    MODEL_PATH,
    TOTAL_FEATURES,
    TRAIN_TEST_SPLIT,
)
from src.data.augmentation import augment_landmarks

warnings.filterwarnings("ignore", category=UserWarning)


class SignClassifier:
    """Trains and runs inference with Random Forest or MLP classifiers."""

    def __init__(self):
        self._model = None
        self._classes = None
        self._model_type = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> bool:
        if not os.path.exists(MODEL_PATH):
            return False
        try:
            data = joblib.load(MODEL_PATH)
            self._model = data["model"]
            self._classes = data["classes"]
            self._model_type = data.get("model_type", "unknown")
            expected = TOTAL_FEATURES
            if hasattr(self._model, "n_features_in_") and self._model.n_features_in_ != expected:
                self._model = None
                return False
            return True
        except Exception:
            self._model = None
            return False

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """Predict sign label and confidence from a feature vector.

        Returns ("", 0.0) if model not loaded or confidence too low.
        """
        if self._model is None:
            return "", 0.0

        features = features.reshape(1, -1)
        probas = self._model.predict_proba(features)[0]
        max_idx = np.argmax(probas)
        confidence = probas[max_idx]

        if confidence < MIN_CONFIDENCE:
            return "", confidence

        return self._classes[max_idx], confidence

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "random_forest",
        progress_callback=None,
    ) -> dict:
        """Train classifier and return evaluation metrics.

        Args:
            X: Feature matrix (n_samples, 84)
            y: Label array
            model_type: "random_forest" or "mlp"
            progress_callback: Optional callable(percent, message)

        Returns dict with keys:
            success, accuracy, cv_accuracy, cv_std,
            classification_report, confusion_matrix_path,
            model_type, n_samples, n_classes, per_class_counts
        """
        result = {
            "success": False,
            "accuracy": 0.0,
            "cv_accuracy": 0.0,
            "cv_std": 0.0,
            "classification_report": "",
            "confusion_matrix_path": "",
            "model_type": model_type,
            "n_samples": len(X),
            "n_classes": len(np.unique(y)),
            "per_class_counts": {},
        }

        if progress_callback:
            progress_callback(5, "Preparing data...")

        classes, counts = np.unique(y, return_counts=True)
        result["per_class_counts"] = dict(zip(classes.tolist(), counts.tolist()))

        # Augment training data
        if progress_callback:
            progress_callback(10, "Augmenting data...")
        X_aug, y_aug = augment_landmarks(X, y)

        # Train/test split
        if progress_callback:
            progress_callback(20, "Splitting train/test data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_aug, y_aug,
            test_size=TRAIN_TEST_SPLIT,
            stratify=y_aug,
            random_state=42,
        )

        # Build model
        if progress_callback:
            progress_callback(30, f"Building {model_type} model...")
        model = self._build_model(model_type, len(classes))

        # Cross-validation on training set
        if progress_callback:
            progress_callback(40, "Running cross-validation...")
        try:
            skf = StratifiedKFold(n_splits=min(CV_FOLDS, min(counts)), shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
            result["cv_accuracy"] = float(np.mean(cv_scores))
            result["cv_std"] = float(np.std(cv_scores))
        except ValueError:
            result["cv_accuracy"] = 0.0
            result["cv_std"] = 0.0

        # Train on full training set
        if progress_callback:
            progress_callback(60, "Training model...")
        model.fit(X_train, y_train)

        # Evaluate on test set
        if progress_callback:
            progress_callback(80, "Evaluating model...")
        y_pred = model.predict(X_test)
        result["accuracy"] = float(accuracy_score(y_test, y_pred))
        result["classification_report"] = classification_report(
            y_test, y_pred, zero_division=0
        )

        # Confusion matrix
        if progress_callback:
            progress_callback(90, "Generating confusion matrix...")
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        self._save_confusion_matrix(cm, classes)
        result["confusion_matrix_path"] = CONFUSION_MATRIX_PATH

        # Save model
        if progress_callback:
            progress_callback(95, "Saving model...")
        self._model = model
        self._classes = classes.tolist()
        self._model_type = model_type

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(
            {"model": model, "classes": self._classes, "model_type": model_type},
            MODEL_PATH,
        )

        result["success"] = True
        if progress_callback:
            progress_callback(100, "Training complete!")
        return result

    @staticmethod
    def _build_model(model_type: str, n_classes: int):
        if model_type == "mlp":
            hidden = max(64, n_classes * 4)
            return MLPClassifier(
                hidden_layer_sizes=(hidden, hidden // 2),
                activation="relu",
                solver="adam",
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
            )
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )

    @staticmethod
    def _save_confusion_matrix(cm: np.ndarray, labels: np.ndarray):
        os.makedirs(os.path.dirname(CONFUSION_MATRIX_PATH), exist_ok=True)
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), max(6, len(labels) * 0.5)))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
        plt.close(fig)
