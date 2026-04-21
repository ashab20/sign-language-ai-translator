# User Guide

This guide explains how to use the Sign Language AI Translator application for real-time sign language detection and speech output.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Detect Signs Tab](#detect-signs-tab)
3. [Train Model Tab](#train-model-tab)
4. [About Tab](#about-tab)
5. [Keyboard Shortcuts & Tips](#keyboard-shortcuts--tips)
6. [FAQ](#faq)

---

## Getting Started

### Starting the Application

```bash
python main.py
```

The application opens with three tabs:
- **Detect Signs** -- Real-time sign detection and sentence building
- **Train Model** -- Record signs and train the AI model
- **About** -- Project information and quick-start guide

### First-Time Setup

If this is your first time using the app:

1. Go to the **Train Model** tab first
2. Record at least 5-10 signs (see [Training Guide](TRAINING_GUIDE.md))
3. Click **Train Model** to build the AI classifier
4. Go to the **Detect Signs** tab to start using it

---

## Detect Signs Tab

This is the main tab where sign language is translated to text and speech.

### Camera Controls

- **Start Camera**: Activates the webcam and begins detection
- **Stop Camera**: Turns off the camera

### Sign Detection

When the camera is running:
- Show a trained sign to the camera
- The **Detected Sign** panel shows the current prediction
- The **Confidence** bar shows how certain the AI is (green = high, red = low)
- Hold a sign steady for about 1 second to confirm it

### Sentence Builder

- Confirmed signs automatically appear in the sentence area
- Signs are separated by spaces
- The sentence builds up as you show different signs

### Sentence Controls

| Button | Action |
|--------|--------|
| **Speak Sentence** | Reads the current sentence aloud using text-to-speech |
| **Backspace** | Removes the last sign from the sentence |
| **Clear Sentence** | Removes all signs and starts over |

### Tips for Best Detection

- Position your hand clearly in the camera frame
- Ensure good lighting (face a light source, don't have light behind you)
- Hold each sign steady for about 1-2 seconds
- Wait for the previous sign to be confirmed before showing the next one
- Watch the confidence bar -- green means the AI is confident

---

## Train Model Tab

This tab is for recording sign data and training the AI classifier.

### Recording Signs

1. **Start Camera**: Turn on the webcam preview
2. **Enter Sign Name**: Type the name of the sign (e.g., `A`, `HELLO`)
3. **Click Record**: The system captures 30 frames of hand landmarks
4. **Hold Steady**: Keep your sign visible and steady during recording
5. The progress bar shows recording progress

### Data Table

The table shows all recorded signs with:
- **Sign**: The sign name/label
- **Samples**: Number of recorded samples
- **Status**: "Ready" (green, 20+ samples) or "Need X more" (red)

### Managing Data

- **Delete Selected**: Remove a sign's data (select it in the table first)
- **Refresh**: Update the table with current data

### Training

1. Ensure all signs show "Ready" status
2. Choose model type:
   - **Random Forest**: Recommended for most use cases
   - **MLP Neural Net**: Alternative neural network approach
3. Click **Train Model** and wait for completion
4. Review the accuracy report in the results panel

### Training Results

After training, the results panel shows:
- Model type used
- Total samples and number of sign classes
- Test accuracy (on unseen data)
- Cross-validation accuracy (consistency measure)
- Per-sign precision, recall, and F1 scores

---

## About Tab

Shows project information:
- University logo and name
- Project title and description
- Student and supervisor names
- Quick-start instructions
- Version information

### Customization

To customize the About tab for your university:

1. Replace `assets/university_logo.png` with your university's logo (recommended: 120x120 pixels)
2. Edit `config.py` to update:
   - `UNIVERSITY_NAME`
   - `STUDENT_NAME`
   - `SUPERVISOR_NAME`
   - `PROJECT_TITLE`

---

## Keyboard Shortcuts & Tips

### Performance Tips

- Close other camera-using apps before starting
- Use a USB webcam for better quality (optional)
- Keep the camera at a consistent distance during training and detection

### Workflow

The recommended daily workflow:

1. Start the application
2. Go to Detect tab, start camera
3. Show signs to build sentences
4. Click Speak to hear the sentence
5. Click Clear to start a new sentence

### Adding New Signs

You can add new signs at any time:

1. Go to Train tab
2. Record the new sign
3. Click Train Model (re-trains with all signs including new ones)
4. Go back to Detect tab -- the new sign is now recognized

---

## FAQ

### Q: How many signs can the system recognize?

There is no hard limit. The system has been tested with up to 26+ signs (full ASL alphabet). Performance depends on how distinct the signs are from each other.

### Q: Does it work with left-handed signing?

Yes. MediaPipe detects both hands. If you train with your left hand, it will recognize left-handed signs.

### Q: Can I use it without internet?

Yes! Everything runs offline -- the AI model, hand detection, and text-to-speech all work without an internet connection.

### Q: How long does training take?

Training typically takes 5-30 seconds depending on the amount of data and model type. Random Forest is faster than MLP.

### Q: Can I export the trained model?

The trained model is saved at `data/model.pkl`. You can copy this file to use it on another computer running the same application.

### Q: The detection is flickering between signs. What should I do?

This usually means the signs are too similar. Try:
- Recording more samples for the confused signs
- Making the signs more distinct
- Ensuring consistent lighting
