"""Builds sentences from a stream of detected sign predictions."""

from collections import deque

from config import CONFIRM_FRAMES, HAND_GONE_FRAMES, SMOOTH_WINDOW


class SentenceBuilder:
    """Accumulates confirmed sign detections into words and sentences.

    Pipeline:
        1. Raw predictions are smoothed via majority vote over a sliding window.
        2. A sign is "confirmed" when the smoothed label is stable for CONFIRM_FRAMES.
        3. When hands disappear for HAND_GONE_FRAMES, the current sign ends.
        4. Confirmed signs are appended to build a sentence.
    """

    def __init__(self):
        self._window: deque[str] = deque(maxlen=SMOOTH_WINDOW)
        self._current_sign = ""
        self._confirm_count = 0
        self._no_hand_count = 0
        self._last_confirmed = ""
        self._sentence_parts: list[str] = []
        self._sign_just_confirmed = False

    @property
    def sentence(self) -> str:
        return " ".join(self._sentence_parts)

    @property
    def current_sign(self) -> str:
        return self._current_sign

    @property
    def sign_just_confirmed(self) -> bool:
        """True for one frame after a new sign is confirmed."""
        return self._sign_just_confirmed

    def update(self, prediction: str, has_hands: bool) -> str:
        """Feed a new frame's prediction. Returns the smoothed label."""
        self._sign_just_confirmed = False

        if not has_hands:
            self._no_hand_count += 1
            if self._no_hand_count >= HAND_GONE_FRAMES and self._current_sign:
                self._finalize_sign()
            self._window.clear()
            return ""

        self._no_hand_count = 0

        if not prediction:
            self._window.append("")
            return self._get_smoothed()

        self._window.append(prediction)
        smoothed = self._get_smoothed()

        if smoothed and smoothed == self._current_sign:
            self._confirm_count += 1
        elif smoothed:
            self._current_sign = smoothed
            self._confirm_count = 1
        else:
            self._confirm_count = 0

        if (
            self._confirm_count >= CONFIRM_FRAMES
            and self._current_sign
            and self._current_sign != self._last_confirmed
        ):
            self._finalize_sign()

        return smoothed

    def _finalize_sign(self):
        if self._current_sign and self._current_sign != self._last_confirmed:
            if self._current_sign == "SPACE":
                self._sentence_parts.append(" ")
            elif self._current_sign == "DELETE":
                if self._sentence_parts:
                    self._sentence_parts.pop()
            else:
                self._sentence_parts.append(self._current_sign)
            self._last_confirmed = self._current_sign
            self._sign_just_confirmed = True
        self._current_sign = ""
        self._confirm_count = 0

    def _get_smoothed(self) -> str:
        if not self._window:
            return ""
        non_empty = [p for p in self._window if p]
        if not non_empty:
            return ""
        counts: dict[str, int] = {}
        for p in non_empty:
            counts[p] = counts.get(p, 0) + 1
        return max(counts, key=counts.get)

    def clear(self):
        self._window.clear()
        self._current_sign = ""
        self._confirm_count = 0
        self._no_hand_count = 0
        self._last_confirmed = ""
        self._sentence_parts.clear()
        self._sign_just_confirmed = False

    def backspace(self):
        if self._sentence_parts:
            self._sentence_parts.pop()
            self._last_confirmed = ""
