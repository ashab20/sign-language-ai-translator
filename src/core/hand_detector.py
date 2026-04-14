"""MediaPipe hands-only detector with simplified feature extraction."""

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
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self._results = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._results = self.hands.process(rgb)
        return frame

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        if self._results and self._results.multi_hand_landmarks:
            for hand_lms in self._results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 200, 100), thickness=2, circle_radius=3),
                    self.mp_draw.DrawingSpec(color=(0, 150, 255), thickness=2),
                )
        return frame

    def has_hands(self) -> bool:
        return bool(self._results and self._results.multi_hand_landmarks)

    def get_features(self) -> np.ndarray | None:
        """Extract 84-D feature vector from detected hands.

        Returns None if no hands are detected.
        """
        if not self._results or not self._results.multi_hand_landmarks:
            return None

        right = np.zeros(FEATURES_PER_HAND, dtype=np.float64)
        left = np.zeros(FEATURES_PER_HAND, dtype=np.float64)

        handedness_list = self._results.multi_handedness or []

        for idx, hand_lms in enumerate(self._results.multi_hand_landmarks):
            vec = self._normalize_hand(hand_lms)

            label = None
            if idx < len(handedness_list):
                label = handedness_list[idx].classification[0].label

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
    def _normalize_hand(hand_lms) -> np.ndarray:
        """Wrist-centered, scale-normalized (x, y) coordinates."""
        pts = np.array(
            [[lm.x, lm.y] for lm in hand_lms.landmark],
            dtype=np.float64,
        )
        wrist = pts[0]
        centered = pts - wrist
        # Scale by distance from wrist to middle finger MCP (landmark 9)
        scale = np.linalg.norm(centered[9]) + 1e-8
        normalized = centered / scale
        return normalized.flatten()

    def release(self):
        self.hands.close()
