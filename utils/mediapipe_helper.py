import cv2
import mediapipe as mp
import numpy as np

# 21 landmarks × 3 coords per hand; concatenated [right, left] for one person
FEATURES_PER_HAND = 21 * 3
POSE_NUM_LANDMARKS = 33
FEATURES_POSE = POSE_NUM_LANDMARKS * 3
HAND_BLOCK_DIM = FEATURES_PER_HAND * 2

# Nose-centered face keypoints (see _KEY_FACE_INDICES), each scaled by inter-eye distance
FACE_KEY_COUNT = 11
FEATURES_FACE = FACE_KEY_COUNT * 3

# Per camera-labeled hand: index pointing dir + tip offsets vs nose and chest (all / scale)
POINTING_BLOCK_DIM = 9
POINTING_BOTH_DIM = POINTING_BLOCK_DIM * 2

# [pose | R hand | L hand | face keys | R pointing | L pointing]
LANDMARK_VECTOR_DIM = (
    FEATURES_POSE + HAND_BLOCK_DIM + FEATURES_FACE + POINTING_BOTH_DIM
)

# MediaPipe Face Mesh landmark indices (topology-stable subset for head / mouth / jaw)
_KEY_FACE_INDICES = np.array(
    [1, 9, 10, 152, 61, 291, 33, 133, 362, 263, 234], dtype=np.int32
)


class HandTracker:
    """
    MediaPipe Pose + Face Mesh + Hands. Feature layout:
    - pose (99): hip-centered, shoulder-scaled
    - hands (126): wrist-centered R then L
    - face (33): key points relative to nose, / inter-ocular scale
    - pointing ×2 (9 each): unit index direction; (tip−nose)/s; (tip−chest)/s
    """

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.65,
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=False,
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.hand_results = None
        self.pose_results = None
        self.face_results = None

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.face_results = self.face_mesh.process(rgb)
        self.pose_results = self.pose.process(rgb)
        self.hand_results = self.hands.process(rgb)
        return frame

    def draw_landmarks(self, frame):
        if self.face_results and self.face_results.multi_face_landmarks:
            for face_lms in self.face_results.multi_face_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    face_lms,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    None,
                    self.mp_draw.DrawingSpec(color=(80, 220, 120), thickness=1),
                )
        if self.pose_results and self.pose_results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame,
                self.pose_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(100, 180, 255), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(100, 180, 255), thickness=2
                ),
            )
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            for hand_lms in self.hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_lms, self.mp_hands.HAND_CONNECTIONS
                )
                self._draw_index_ray(frame, hand_lms)
        return frame

    @staticmethod
    def _draw_index_ray(frame, hand_lms):
        """Visual cue for pointing direction (wrist → index tip)."""
        lm = hand_lms.landmark
        h, w = frame.shape[0], frame.shape[1]
        p0 = (int(lm[0].x * w), int(lm[0].y * h))
        p1 = (int(lm[8].x * w), int(lm[8].y * h))
        cv2.line(frame, p0, p1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, p1, 5, (0, 255, 255), -1, cv2.LINE_AA)

    @staticmethod
    def _normalize_hand(hand) -> np.ndarray:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float64)
        wrist = pts[0]
        centered = pts - wrist
        scale = np.linalg.norm(centered[9]) + 1e-6
        return (centered / scale).flatten()

    @staticmethod
    def _normalize_pose(pose_landmarks) -> np.ndarray:
        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark], dtype=np.float64
        )
        hip_mid = (pts[23] + pts[24]) * 0.5
        centered = pts - hip_mid
        scale = float(np.linalg.norm(pts[11] - pts[12])) + 1e-6
        return (centered / scale).flatten()

    @staticmethod
    def _pose_xyz_array(pose_landmarks) -> np.ndarray:
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in pose_landmarks.landmark], dtype=np.float64
        )

    @staticmethod
    def _chest_point(pts: np.ndarray) -> np.ndarray:
        """Sternum-ish point between shoulders and hips in normalized image space."""
        shoulder_mid = (pts[11] + pts[12]) * 0.5
        hip_mid = (pts[23] + pts[24]) * 0.5
        return shoulder_mid * 0.62 + hip_mid * 0.38

    @staticmethod
    def _hand_span_scale(hand_lms) -> float:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float64)
        return float(np.linalg.norm(pts[8] - pts[17])) + 1e-6

    def _face_scale_nose(self, face_landmarks) -> tuple[float, np.ndarray]:
        """Inter-ocular distance (iris centers if refined, else outer eye corners) and nose tip."""
        lm = face_landmarks.landmark
        n = len(lm)
        nose = np.array([lm[1].x, lm[1].y, lm[1].z], dtype=np.float64)
        if n > 473:
            le = np.array([lm[468].x, lm[468].y, lm[468].z], dtype=np.float64)
            re = np.array([lm[473].x, lm[473].y, lm[473].z], dtype=np.float64)
            dist = float(np.linalg.norm(le - re))
        else:
            p33 = np.array([lm[33].x, lm[33].y, lm[33].z], dtype=np.float64)
            p263 = np.array([lm[263].x, lm[263].y, lm[263].z], dtype=np.float64)
            dist = float(np.linalg.norm(p33 - p263))
        return max(dist, 1e-4), nose

    def _face_embedding(self, face_landmarks) -> np.ndarray:
        lm = face_landmarks.landmark
        scale, nose = self._face_scale_nose(face_landmarks)
        out = np.zeros(FEATURES_FACE, dtype=np.float64)
        k = 0
        for idx in _KEY_FACE_INDICES:
            p = np.array([lm[int(idx)].x, lm[int(idx)].y, lm[int(idx)].z], dtype=np.float64)
            out[k : k + 3] = (p - nose) / scale
            k += 3
        return out

    @staticmethod
    def _pointing_block(
        hand_lms,
        scale: float,
        nose_xyz: np.ndarray | None,
        chest_xyz: np.ndarray | None,
    ) -> np.ndarray:
        """Unit index ray; (tip−nose)/s; (tip−chest)/s (chest from pose torso blend)."""
        block = np.zeros(POINTING_BLOCK_DIM, dtype=np.float64)
        if scale < 1e-9:
            return block
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float64)
        wrist, tip = pts[0], pts[8]
        d = tip - wrist
        block[0:3] = d / (np.linalg.norm(d) + 1e-9)
        if nose_xyz is not None:
            block[3:6] = (tip - nose_xyz) / scale
        if chest_xyz is not None:
            block[6:9] = (tip - chest_xyz) / scale
        return block

    def get_landmarks(self):
        pose_vec = np.zeros(FEATURES_POSE, dtype=np.float64)
        right = np.zeros(FEATURES_PER_HAND, dtype=np.float64)
        left = np.zeros(FEATURES_PER_HAND, dtype=np.float64)
        face_vec = np.zeros(FEATURES_FACE, dtype=np.float64)
        point_r = np.zeros(POINTING_BLOCK_DIM, dtype=np.float64)
        point_l = np.zeros(POINTING_BLOCK_DIM, dtype=np.float64)

        has_pose = False
        pose_pts = None
        if self.pose_results and self.pose_results.pose_landmarks:
            pose_vec = self._normalize_pose(self.pose_results.pose_landmarks)
            pose_pts = self._pose_xyz_array(self.pose_results.pose_landmarks)
            has_pose = True

        nose_xyz = None
        face_scale = 0.0
        if self.face_results and self.face_results.multi_face_landmarks:
            fl = self.face_results.multi_face_landmarks[0]
            face_vec = self._face_embedding(fl)
            face_scale, nose_xyz = self._face_scale_nose(fl)

        chest_xyz = None
        shoulder_scale = 0.0
        if pose_pts is not None:
            chest_xyz = self._chest_point(pose_pts)
            shoulder_scale = float(np.linalg.norm(pose_pts[11] - pose_pts[12])) + 1e-6

        scale = max(face_scale, shoulder_scale, 0.08)

        has_hand = False
        if self.hand_results and self.hand_results.multi_hand_landmarks:
            has_hand = True
            handedness_list = self.hand_results.multi_handedness or []
            for idx, hand_lms in enumerate(self.hand_results.multi_hand_landmarks):
                vec = self._normalize_hand(hand_lms)
                span = self._hand_span_scale(hand_lms)
                s = max(scale, span)
                pb = self._pointing_block(hand_lms, s, nose_xyz, chest_xyz)
                label = None
                if idx < len(handedness_list):
                    label = handedness_list[idx].classification[0].label
                if label == "Right":
                    right = vec
                    point_r = pb
                elif label == "Left":
                    left = vec
                    point_l = pb
                else:
                    if idx == 0:
                        right = vec
                        point_r = pb
                    else:
                        left = vec
                        point_l = pb

        if not has_pose and not has_hand:
            return None

        hand_vec = np.concatenate([right, left])
        point_vec = np.concatenate([point_r, point_l])
        return np.concatenate([pose_vec, hand_vec, face_vec, point_vec])
