"""Data augmentation for landmark feature vectors."""

import numpy as np

from config import AUGMENTATION_FACTOR, AUGMENTATION_NOISE_STD


def augment_landmarks(
    X: np.ndarray,
    y: np.ndarray,
    factor: int = AUGMENTATION_FACTOR,
    noise_std: float = AUGMENTATION_NOISE_STD,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment training data by adding Gaussian noise to landmark features.

    For each original sample, creates (factor - 1) noisy copies.
    The original samples are preserved unchanged.

    Returns augmented (X, y) arrays.
    """
    augmented_X = [X]
    augmented_y = [y]

    for _ in range(factor - 1):
        noise = np.random.normal(0, noise_std, X.shape)
        noisy_X = X + noise
        augmented_X.append(noisy_X)
        augmented_y.append(y.copy())

    return np.vstack(augmented_X), np.concatenate(augmented_y)


def augment_single_sample(
    landmarks: np.ndarray,
    count: int = 5,
    noise_std: float = AUGMENTATION_NOISE_STD,
) -> list[np.ndarray]:
    """Create augmented copies of a single landmark vector."""
    results = []
    for _ in range(count):
        noise = np.random.normal(0, noise_std, landmarks.shape)
        results.append(landmarks + noise)
    return results
