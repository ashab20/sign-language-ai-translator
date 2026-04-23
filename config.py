"""Application-wide configuration constants."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# About tab: searched in order under assets/ (put your logo as ju.png or ju.svg)
UNIVERSITY_LOGO_RASTER_FILES = ("ju.png", "ju.jpg", "ju.jpeg", "university_logo.png")
UNIVERSITY_LOGO_SVG_FILES = ("ju.svg", "university_logo.svg")

DB_PATH = os.path.join(DATA_DIR, "gestures.db")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")
CONFUSION_MATRIX_PATH = os.path.join(DATA_DIR, "confusion_matrix.png")

# MediaPipe hand landmarks: 21 landmarks x 2 coords (x, y) x 2 hands
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 2  # x, y only (drop z for stability)
FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK
TOTAL_FEATURES = FEATURES_PER_HAND * 2  # both hands = 84

# Training
MIN_SAMPLES_PER_SIGN = 20
RECORDING_FRAMES = 30
TRAIN_TEST_SPLIT = 0.2
CV_FOLDS = 5
AUGMENTATION_FACTOR = 3  # multiply training data by this factor
AUGMENTATION_NOISE_STD = 0.02

# Detection / smoothing
SMOOTH_WINDOW = 10
CONFIRM_FRAMES = 8
HAND_GONE_FRAMES = 15
MIN_CONFIDENCE = 0.4

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# UI
APP_TITLE = "Sign Language AI Translator"
WINDOW_SIZE = "1100x750"
UNIVERSITY_NAME = "Jahangirnagar University"
PROJECT_TITLE = "Sign Language to Speech Converter"
PROJECT_SUBTITLE = "AI-Powered ASL Recognition System"
STUDENT_NAME = "Ahmad Ashab Uddin, Md Adnan Hossain, Aminul Islam"
SUPERVISOR_NAME = "Dr. Shamim Al Mamun"
