"""MediaPipe hands detector with feature extraction.

Supports both:
- legacy `mp.solutions.hands` API
- modern MediaPipe Tasks API (`HandLandmarker`)
"""

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np

from config import FEATURES_PER_HAND, TOTAL_FEATURES


class HandDetector:
    """Detects hands via MediaPipe and extracts a normalized feature vector.

    Feature layout: [right_hand (42 floats), left_hand (42 floats)] = 84 total.
    Each hand: 21 landmarks x 2 coords (x, y), wrist-centered, scale-normalized.
    """

    def __init__(self):
        self._using_legacy = hasattr(mp, "solutions")
        self._task_connections = None
        self._task_result = None

        if self._using_legacy:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6,
            )
        else:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model_path = self._ensure_task_model()
            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                num_hands=2,
            )
            self.hands = vision.HandLandmarker.create_from_options(options)
            self._task_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        self._results = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._using_legacy:
            self._results = self.hands.process(rgb)
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self._task_result = self.hands.detect(mp_image)
            self._results = self._task_result
        return frame

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        if not self._results:
            return frame

        if self._using_legacy and self._results.multi_hand_landmarks:
            for hand_lms in self._results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 200, 100), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(0, 150, 255), thickness=2),
                )
        elif not self._using_legacy and self._task_result and self._task_result.hand_landmarks:
            h, w = frame.shape[:2]
            for hand_lms in self._task_result.hand_landmarks:
                points = [
                    (int(lm.x * w), int(lm.y * h))
                    for lm in hand_lms
                ]
                for conn in self._task_connections:
                    start = points[conn.start]
                    end = points[conn.end]
                    cv2.line(frame, start, end, (0, 150, 255), 2)
                for x, y in points:
                    cv2.circle(frame, (x, y), 3, (0, 200, 100), -1)
        return frame

    def has_hands(self) -> bool:
        if not self._results:
            return False
        if self._using_legacy:
            return bool(self._results.multi_hand_landmarks)
        return bool(self._task_result and self._task_result.hand_landmarks)

    def get_features(self) -> np.ndarray | None:
        """Extract 84-D feature vector from detected hands.

        Returns None if no hands are detected.
        """
        if not self._results:
            return None

        right = np.zeros(FEATURES_PER_HAND, dtype=np.float64)
        left = np.zeros(FEATURES_PER_HAND, dtype=np.float64)

        if self._using_legacy:
            hand_landmarks = self._results.multi_hand_landmarks or []
            handedness_list = self._results.multi_handedness or []
        else:
            hand_landmarks = self._task_result.hand_landmarks or []
            handedness_list = self._task_result.handedness or []

        if not hand_landmarks:
            return None

        for idx, hand_lms in enumerate(hand_landmarks):
            vec = self._normalize_hand(hand_lms, self._using_legacy)

            label = None
            if idx < len(handedness_list):
                if self._using_legacy:
                    label = handedness_list[idx].classification[0].label
                else:
                    label = handedness_list[idx][0].category_name

            if label == "Right":
                right = vec
            elif label == "Left":
                left = vec
            elif idx == 0:
                right = vec
            else:
                left = vec

        return np.concatenate([right, left])

    @staticmethod
    def _normalize_hand(hand_lms, using_legacy: bool) -> np.ndarray:
        """Wrist-centered, scale-normalized (x, y) coordinates."""
        if using_legacy:
            pts = np.array([[lm.x, lm.y] for lm in hand_lms.landmark], dtype=np.float64)
        else:
            pts = np.array([[lm.x, lm.y] for lm in hand_lms], dtype=np.float64)
        wrist = pts[0]
        centered = pts - wrist
        # Scale by distance from wrist to middle finger MCP (landmark 9)
        scale = np.linalg.norm(centered[9]) + 1e-8
        normalized = centered / scale
        return normalized.flatten()

    def release(self):
        self.hands.close()

    @staticmethod
    def _ensure_task_model() -> Path:
        model_path = Path(".cache") / "mediapipe" / "hand_landmarker.task"
        if model_path.exists():
            return model_path

        model_path.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task",
            model_path,
        )
        return model_path
