import cv2
import mediapipe as mp
import numpy as np

# 21 landmarks × 3 coords per hand; concatenated [right, left] for one person
FEATURES_PER_HAND = 21 * 3
LANDMARK_VECTOR_DIM = FEATURES_PER_HAND * 2


class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.65,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        return frame

    def draw_landmarks(self, frame):
        if self.results and self.results.multi_hand_landmarks:
            for hand_lms in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_lms, self.mp_hands.HAND_CONNECTIONS
                )
        return frame

    @staticmethod
    def _normalize_landmarks(hand) -> np.ndarray:
        """Wrist-relative + scale so pose works across distance and position."""
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float64)
        wrist = pts[0]
        centered = pts - wrist
        scale = np.linalg.norm(centered[9]) + 1e-6
        return (centered / scale).flatten()

    def get_landmarks(self):
        """
        Both hands for the same person: 63 floats for camera-Right hand, then 63 for
        camera-Left (MediaPipe labels). Missing hand is zeros so one-handed signs still work.
        """
        if not self.results or not self.results.multi_hand_landmarks:
            return None

        right = np.zeros(FEATURES_PER_HAND, dtype=np.float64)
        left = np.zeros(FEATURES_PER_HAND, dtype=np.float64)
        handedness_list = self.results.multi_handedness or []

        for idx, hand_lms in enumerate(self.results.multi_hand_landmarks):
            vec = self._normalize_landmarks(hand_lms)
            label = None
            if idx < len(handedness_list):
                label = handedness_list[idx].classification[0].label
            if label == "Right":
                right = vec
            elif label == "Left":
                left = vec
            else:
                # Rare fallback if handedness is missing
                if idx == 0:
                    right = vec
                else:
                    left = vec

        return np.concatenate([right, left])
