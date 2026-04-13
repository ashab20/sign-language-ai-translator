import tkinter as tk
import ttkbootstrap as tbs
from ttkbootstrap.constants import *
from typing import Callable

from services.predictor import IGNORE_PREDICTIONS
from utils.camera_permission import request_camera_permission

# Match video frames drawn in ml_model / controller (480×360 RGB)
CAM_W, CAM_H = 480, 360

_SENTENCE_PLACEHOLDER = "(Completed signs appear here as a sentence — lower hands after each sign.)"


class _CollapsibleSection:
    """Single-column disclosure: click header to show or hide body."""

    def __init__(
        self,
        parent,
        title: str,
        *,
        start_open: bool = False,
        pady: tuple[int, int] = (0, 6),
    ):
        self._open = start_open
        self._title = title
        self.outer = tbs.Frame(parent)
        self.outer.pack(fill=X, pady=pady)
        self.toggle_btn = tbs.Button(
            self.outer,
            text=self._button_label(),
            command=self.toggle,
            bootstyle=SECONDARY,
        )
        self.toggle_btn.pack(anchor=W, fill=X)
        self.content = tbs.Frame(self.outer)
        if start_open:
            self.content.pack(fill=X, pady=(6, 0), padx=(6, 0))

    def _button_label(self) -> str:
        return ("▼ " if self._open else "▶ ") + self._title

    def toggle(self) -> None:
        self._open = not self._open
        self.toggle_btn.configure(text=self._button_label())
        if self._open:
            self.content.pack(fill=X, pady=(6, 0), padx=(6, 0))
        else:
            self.content.pack_forget()


class MainView:
    """Main UI — camera on the left, controls on the right."""

    def __init__(self, controller):
        self.controller = controller

        self.root = tbs.Window(themename="darkly")
        self.root.withdraw()
        self.root.title("Sign Language AI Translator")
        self.root.geometry("1080x780")
        self.root.minsize(920, 640)
        self.root.resizable(True, True)

        if not request_camera_permission(parent=self.root):
            self.root.destroy()
            return

        self.speak_on_end = tbs.BooleanVar(value=True)
        self.auto_start_live = tbs.BooleanVar(value=True)

        self._build_ui()
        self.root.deiconify()

    def _build_ui(self):
        header = tbs.Frame(self.root, padding=(16, 12))
        header.pack(fill=X)
        tbs.Label(
            header,
            text="Sign Language AI Translator",
            font=("Helvetica Neue", 20, "bold"),
            bootstyle=PRIMARY,
        ).pack(anchor=W)
        tbs.Label(
            header,
            text="Camera on the left — use collapsible sections on the right to save space.",
            font=("Helvetica Neue", 11),
            bootstyle=SECONDARY,
        ).pack(anchor=W, pady=(4, 0))

        body = tbs.Frame(self.root, padding=(14, 8))
        body.pack(fill=BOTH, expand=True)

        # ——— Left: live camera ———
        left_col = tbs.Frame(body, padding=(0, 0, 12, 0))
        left_col.pack(side=LEFT, fill=Y, expand=False, anchor=N)

        cam_frame = tbs.Labelframe(left_col, text="Live camera", padding=10)
        cam_frame.pack(fill=BOTH, expand=True)

        self.cam_canvas = tk.Canvas(
            cam_frame,
            width=CAM_W,
            height=CAM_H,
            bg="#1a1a1f",
            highlightthickness=1,
            highlightbackground="#333",
        )
        self.cam_canvas.pack()

        def keep_ref(img: tk.PhotoImage):
            self.cam_canvas.delete("all")
            self.cam_canvas.create_image(
                CAM_W // 2, CAM_H // 2, image=img, anchor="center"
            )
            self.cam_canvas.image = img

        self.draw_on_canvas: Callable[[tk.PhotoImage], None] = keep_ref

        # ——— Right column ———
        right_col = tbs.Frame(body)
        right_col.pack(side=LEFT, fill=BOTH, expand=True, anchor=N)

        # ——— Always visible: sentence + live controls ———
        sentence_frame = tbs.Labelframe(
            right_col, text="Signed sentence (live communicate)", padding=(12, 10)
        )
        sentence_frame.pack(fill=BOTH, expand=True, pady=(0, 8))

        tbs.Label(
            sentence_frame,
            text="Start live, then sign each word and lower your hands to add it to the line below.",
            font=("Helvetica Neue", 10),
            bootstyle=SECONDARY,
            wraplength=520,
        ).pack(anchor=W, pady=(0, 6))

        st_row = tbs.Frame(sentence_frame)
        st_row.pack(fill=BOTH, expand=True, pady=(0, 8))
        st_sb = tbs.Scrollbar(st_row, bootstyle=ROUND)
        self.sentence_text = tk.Text(
            st_row,
            height=6,
            font=("Consolas", 12),
            wrap=tk.WORD,
            yscrollcommand=st_sb.set,
            bg="#2b2b2b",
            fg="#e8e8e8",
            insertbackground="#e8e8e8",
            highlightthickness=1,
            highlightbackground="#444",
            borderwidth=0,
        )
        st_sb.config(command=self.sentence_text.yview)
        self.sentence_text.pack(side=LEFT, fill=BOTH, expand=True)
        st_sb.pack(side=RIGHT, fill=Y)

        self.sentence_text.insert("1.0", _SENTENCE_PLACEHOLDER)
        self.sentence_text.configure(state="disabled")

        sent_btns = tbs.Frame(sentence_frame)
        sent_btns.pack(fill=X)
        tbs.Button(
            sent_btns,
            text="Clear sentence",
            bootstyle=SECONDARY,
            width=14,
            command=self.controller.clear_sentence,
        ).pack(side=LEFT, padx=(0, 8))
        tbs.Button(
            sent_btns,
            text="Read sentence aloud",
            bootstyle=INFO,
            width=18,
            command=self.controller.speak_sentence,
        ).pack(side=LEFT, padx=2)

        live_row = tbs.Frame(sentence_frame)
        live_row.pack(fill=X, pady=(10, 0))
        tbs.Button(
            live_row,
            text="Start live",
            bootstyle=PRIMARY,
            width=14,
            command=self.controller.use_ai,
        ).pack(side=LEFT, padx=(0, 6))
        tbs.Button(
            live_row,
            text="Stop AI",
            bootstyle=DANGER,
            width=14,
            command=self.controller.stop_ai,
        ).pack(side=LEFT, padx=6)

        # ——— Collapsible: Recognition ———
        rec_sec = _CollapsibleSection(right_col, "Recognition", start_open=True)
        pred_outer = rec_sec.content

        tbs.Label(
            pred_outer,
            text="Live (smoothed)",
            font=("Helvetica Neue", 11, "bold"),
        ).pack(anchor=W)
        self.pred_var = tbs.StringVar(value="—")
        tbs.Entry(
            pred_outer,
            textvariable=self.pred_var,
            font=("Consolas", 14),
            state="readonly",
            bootstyle=INFO,
        ).pack(fill=X, pady=(4, 8))

        tbs.Label(
            pred_outer,
            text="Last completed sign",
            font=("Helvetica Neue", 11, "bold"),
        ).pack(anchor=W)
        self.last_completed_var = tbs.StringVar(value="—")
        tbs.Entry(
            pred_outer,
            textvariable=self.last_completed_var,
            font=("Consolas", 13),
            state="readonly",
            bootstyle=SECONDARY,
        ).pack(fill=X, pady=(4, 0))

        tbs.Checkbutton(
            pred_outer,
            text="Read aloud when a sign ends",
            variable=self.speak_on_end,
            bootstyle="round-toggle",
        ).pack(anchor=W, pady=(10, 0))

        tbs.Checkbutton(
            pred_outer,
            text="Auto-start live on open",
            variable=self.auto_start_live,
            bootstyle="round-toggle",
        ).pack(anchor=W, pady=(6, 0))

        # ——— Collapsible: Train & record ———
        train_sec = _CollapsibleSection(right_col, "Train & record", start_open=False)
        actions = train_sec.content

        row1 = tbs.Frame(actions)
        row1.pack(fill=X, pady=(0, 6))
        tbs.Button(
            row1,
            text="Record gesture",
            bootstyle=WARNING,
            width=18,
            command=self.controller.record_gesture,
        ).pack(side=LEFT, padx=(0, 6))
        tbs.Button(
            row1,
            text="Train model",
            bootstyle=SUCCESS,
            width=16,
            command=self.controller.train_model,
        ).pack(side=LEFT, padx=6)

        # ——— Collapsible: Text → sign animation ———
        anim_sec = _CollapsibleSection(
            right_col, "Text → sign animation", start_open=False
        )
        anim = anim_sec.content

        self.text_var = tbs.StringVar(value="HELLO WORLD")
        tbs.Entry(
            anim,
            textvariable=self.text_var,
            font=("Consolas", 12),
        ).pack(fill=X, pady=(0, 8))

        anim_btns = tbs.Frame(anim)
        anim_btns.pack(fill=X)
        tbs.Button(
            anim_btns,
            text="Play live",
            bootstyle=INFO,
            width=12,
            command=lambda: self.controller.play_text(record=False),
        ).pack(side=LEFT, padx=(0, 6))
        tbs.Button(
            anim_btns,
            text="Record video",
            bootstyle=DANGER,
            width=14,
            command=lambda: self.controller.play_text(record=True),
        ).pack(side=LEFT, padx=6)
        tbs.Button(
            anim_btns,
            text="Play last",
            bootstyle=SECONDARY,
            width=12,
            command=self.controller.play_last_video,
        ).pack(side=RIGHT, padx=6)

        self.status_var = tbs.StringVar(value="Ready")
        tbs.Label(
            self.root,
            textvariable=self.status_var,
            relief=SUNKEN,
            anchor=W,
            padding=(10, 8),
            bootstyle=SECONDARY,
        ).pack(fill=X, side=BOTTOM)

    def update_prediction(self, sign: str) -> None:
        self.pred_var.set(sign.upper() if sign else "—")

    def update_last_completed(self, sign: str) -> None:
        self.last_completed_var.set(sign.upper() if sign else "—")

    def update_status(self, msg: str) -> None:
        self.status_var.set(msg)

    def append_sentence(self, token: str) -> None:
        t = (token or "").strip().upper()
        if not t or t in IGNORE_PREDICTIONS:
            return
        self.sentence_text.configure(state="normal")
        cur = self.sentence_text.get("1.0", "end").strip()
        if cur == _SENTENCE_PLACEHOLDER or not cur:
            self.sentence_text.delete("1.0", "end")
            self.sentence_text.insert("1.0", t)
        else:
            self.sentence_text.insert(tk.END, " " + t)
        self.sentence_text.configure(state="disabled")
        self.sentence_text.see(tk.END)

    def clear_sentence_text(self) -> None:
        self.sentence_text.configure(state="normal")
        self.sentence_text.delete("1.0", "end")
        self.sentence_text.insert("1.0", _SENTENCE_PLACEHOLDER)
        self.sentence_text.configure(state="disabled")

    def get_sentence_text(self) -> str:
        raw = self.sentence_text.get("1.0", "end").strip()
        if raw == _SENTENCE_PLACEHOLDER:
            return ""
        return raw

    # Backwards-compatible names for any older callers
    def append_translation(self, token: str) -> None:
        self.append_sentence(token)

    def clear_translation_text(self) -> None:
        self.clear_sentence_text()

    def get_translation_text(self) -> str:
        return self.get_sentence_text()

    def run(self) -> None:
        self.root.mainloop()
