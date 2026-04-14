# Sign Language AI Translator

An AI-powered desktop application that translates American Sign Language (ASL) gestures into text and speech in real-time. Built as a university thesis project.

## Overview

This application uses computer vision (MediaPipe) and machine learning (Random Forest / MLP Neural Network) to:

1. **Detect** hand signs through a webcam in real-time
2. **Classify** signs into letters, words, or phrases
3. **Build sentences** from a sequence of detected signs
4. **Speak** the sentence aloud using text-to-speech

The goal is to help people who communicate through sign language be understood by everyone.

## Features

- Real-time hand tracking and sign detection via webcam
- Easy training interface -- record your own signs with a click
- Random Forest and MLP Neural Network classifiers
- Data augmentation for improved accuracy
- Training evaluation with accuracy reports and confusion matrix
- Sentence builder with automatic sign confirmation
- Text-to-speech output (offline, no internet required)
- Clean, tabbed user interface

## Quick Start

### 1. Install Python

Requires **Python 3.11 or later**. Download from [python.org](https://www.python.org/downloads/).

### 2. Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv env
source env/bin/activate    # macOS/Linux
# env\Scripts\activate     # Windows

# Install packages
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python main.py
```

### 4. Train the Model

1. Go to the **Train Model** tab
2. Start the camera
3. Enter a sign name (e.g., `A`, `HELLO`, `THANK YOU`)
4. Click **Record** and hold your sign steady for 30 frames
5. Repeat for all signs (minimum 20 samples per sign)
6. Click **Train Model**
7. Review accuracy results

### 5. Start Detecting

1. Go to the **Detect Signs** tab
2. Start the camera
3. Show signs to the camera -- detected signs build into a sentence
4. Click **Speak Sentence** to hear the sentence read aloud

## Project Structure

```
sign-language-ai-translator/
├── main.py                  # Application entry point
├── config.py                # Configuration constants
├── requirements.txt         # Python dependencies
├── assets/                  # University logo and images
├── data/                    # Database and trained model (auto-created)
└── src/
    ├── ui/                  # User interface (3 tabs)
    │   ├── app_window.py    # Main window
    │   ├── detect_tab.py    # Sign detection tab
    │   ├── train_tab.py     # Training tab
    │   └── about_tab.py     # About/info tab
    ├── core/                # Core logic
    │   ├── hand_detector.py # MediaPipe hand detection
    │   ├── classifier.py    # ML model (Random Forest / MLP)
    │   ├── sentence_builder.py
    │   └── tts_engine.py    # Text-to-speech
    └── data/                # Data layer
        ├── database.py      # SQLite database operations
        └── augmentation.py  # Training data augmentation
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Hand Tracking | MediaPipe Hands |
| ML Classifier | scikit-learn (Random Forest, MLP) |
| Database | SQLite + SQLAlchemy |
| GUI | tkinter + ttkbootstrap |
| Text-to-Speech | pyttsx3 |
| Visualization | matplotlib + seaborn |

## How It Works

1. **Hand Detection**: MediaPipe detects up to 2 hands and extracts 21 landmarks per hand
2. **Feature Extraction**: Landmarks are normalized (wrist-centered, scale-invariant) into an 84-dimensional feature vector
3. **Classification**: A trained Random Forest or MLP model predicts the sign label
4. **Smoothing**: A rolling window + majority vote prevents flickering between predictions
5. **Sentence Building**: Confirmed signs are accumulated into a sentence
6. **Text-to-Speech**: The sentence is spoken using pyttsx3

## Documentation

- [Training Guide](TRAINING_GUIDE.md) -- Step-by-step instructions for training the AI
- [User Guide](USER_GUIDE.md) -- How to use the application

## Compatibility

| OS | Status |
|----|--------|
| macOS (Intel + Apple Silicon) | Supported |
| Windows 10/11 | Supported |
| Linux (Ubuntu 20.04+) | Supported |

## License

This project is part of a university thesis and is intended for educational purposes.
