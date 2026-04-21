# Training Guide

This guide explains how to train the Sign Language AI Translator to recognize your signs with high accuracy.

## Table of Contents

1. [Before You Start](#before-you-start)
2. [Recording Signs](#recording-signs)
3. [Training the Model](#training-the-model)
4. [Understanding Results](#understanding-results)
5. [Tips for High Accuracy](#tips-for-high-accuracy)
6. [Troubleshooting](#troubleshooting)

---

## Before You Start

### Environment Setup

- **Lighting**: Use a well-lit room. Avoid backlighting (don't sit in front of a window).
- **Background**: A plain, non-cluttered background works best.
- **Camera Position**: Position the camera at chest height, about arm's length away.
- **Clothing**: Avoid gloves. Long sleeves that cover the wrist can reduce accuracy.

### Planning Your Signs

Decide which signs you want to train. Common options:

| Category | Examples |
|----------|---------|
| ASL Alphabet | A, B, C, D, ... Z |
| Numbers | 1, 2, 3, 4, 5 |
| Common Words | HELLO, THANK YOU, YES, NO, PLEASE |
| Phrases | GOOD MORNING, HOW ARE YOU |

> **Note**: Start with 5-10 signs to test. You can always add more later.

---

## Recording Signs

### Step-by-Step

1. Open the application and go to the **Train Model** tab
2. Click **Start Camera** to activate the webcam
3. Type the sign name in the text field (e.g., `A` or `HELLO`)
4. Position your hand in the sign gesture within the camera view
5. Click **Record (30 frames)** and hold your sign steady
6. Wait for the progress bar to complete
7. The status will show "Saved X samples for 'SIGN_NAME'"

### Recording Best Practices

- **Hold steady**: Keep your hand as still as possible during recording
- **Multiple sessions**: Record the same sign 2-3 times (click Record again with the same name). This adds variety to your training data.
- **Slight variations**: Between recording sessions, slightly vary your hand angle and position. This helps the AI generalize.
- **Minimum samples**: You need at least **20 samples per sign** (1 recording session = 30 frames, so one session is enough)
- **Recommended samples**: For best results, aim for **60-90 samples per sign** (2-3 recording sessions)
- **Both hands**: If you want to support both left and right-handed signing, record with both hands

### Sample Count Guide

| Samples per Sign | Expected Quality |
|-----------------|------------------|
| 20-30 | Minimum viable (may have some errors) |
| 60-90 | Good accuracy for most signs |
| 120+ | Excellent accuracy, handles variation well |

---

## Training the Model

### Step-by-Step

1. Ensure all signs in the data table show **"Ready"** status (green)
2. Choose a model type:
   - **Random Forest** (recommended) -- fast, reliable, good for most cases
   - **MLP Neural Net** -- neural network approach, may capture complex patterns
3. Click **Train Model**
4. Wait for training to complete (watch the progress bar)
5. Review the accuracy report

### What Happens During Training

The training pipeline performs these steps automatically:

1. **Data Augmentation**: Creates 3x more training samples by adding small noise variations
2. **Train/Test Split**: Reserves 20% of data for testing (never seen during training)
3. **Cross-Validation**: Tests the model 5 different ways to ensure consistency
4. **Model Training**: Trains the final classifier on the full training set
5. **Evaluation**: Tests accuracy on the reserved test data
6. **Confusion Matrix**: Generates a visual chart showing per-sign accuracy

---

## Understanding Results

### Accuracy Metrics

After training, you'll see:

- **Test Accuracy**: How well the model performs on data it has never seen. Target: **90%+**
- **CV Accuracy**: Average accuracy across 5 cross-validation folds. Shows consistency.
- **Per-Class Precision/Recall/F1**: Individual performance for each sign

### Reading the Classification Report

```
              precision    recall  f1-score   support
           A       0.95      0.98      0.96        20
           B       0.92      0.90      0.91        20
       HELLO       1.00      0.95      0.97        20
```

- **Precision**: Of all predictions for this sign, how many were correct?
- **Recall**: Of all actual instances of this sign, how many were detected?
- **F1-Score**: Combined measure (higher is better, max 1.00)
- **Support**: Number of test samples

### Confusion Matrix

A confusion matrix image is saved at `data/confusion_matrix.png`. It shows:
- Diagonal cells (dark blue) = correct predictions
- Off-diagonal cells = misclassifications (which signs get confused with each other)

---

## Tips for High Accuracy

### Do's

1. Record in the same environment you'll use for detection
2. Include slight hand position variations across recording sessions
3. Keep similar lighting between training and detection
4. Train with at least 60 samples per sign for reliable results
5. Re-train after adding new signs (the model includes all signs)

### Don'ts

1. Don't record with extreme angles or partial hand visibility
2. Don't mix left and right hand recordings unless you want both-hand support
3. Don't train with too few signs initially -- start with 5-10 and add gradually
4. Don't ignore low-accuracy signs -- delete and re-record them

### Fixing Low Accuracy

If a specific sign has low accuracy:

1. Check the confusion matrix to see which signs it's confused with
2. Delete that sign's data from the table
3. Re-record with clearer hand positioning
4. Ensure the confused signs are visually distinct
5. Re-train the model

---

## Troubleshooting

### "Not enough samples" Error

You need at least 20 samples per sign. Record more sessions for that sign.

### "No valid samples" Error

The feature size doesn't match. This can happen if the code was updated. Delete old data and re-record.

### Low Overall Accuracy (<80%)

- Record more samples per sign (aim for 60+)
- Check lighting conditions
- Ensure signs are visually distinct from each other
- Try the MLP model instead of Random Forest (or vice versa)

### Signs Keep Getting Confused

Two signs may look too similar. Solutions:
- Exaggerate the differences when recording
- Remove one of the similar signs
- Add more samples with clear distinctions

### Camera Not Detected

- Check that no other application is using the camera
- On macOS, grant camera permission when prompted
- Try changing `CAMERA_INDEX` in `config.py` (0, 1, or 2)
