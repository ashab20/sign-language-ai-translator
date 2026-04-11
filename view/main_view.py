import tkinter as tk
import ttkbootstrap as tbs
from ttkbootstrap.constants import *
from typing import Callable

try:
    from ttkbootstrap.scrolled import ScrolledText as TranscriptText
except ImportError:
    TranscriptText = None

from services.predictor import IGNORE_PREDICTIONS
from utils.camera_permission import request_camera_permission

# Match video frames drawn in ml_model / controller (480×360 RGB)
CAM_W, CAM_H = 480, 360


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
            text="Camera on the left — recognition, translation, and actions on the right.",
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

        # ——— Right: all other UI ———
        right_col = tbs.Frame(body)
        right_col.pack(side=LEFT, fill=BOTH, expand=True, anchor=N)

        pred_outer = tbs.Labelframe(right_col, text="Recognition", padding=(12, 10))
        pred_outer.pack(fill=X, pady=(0, 8))

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
            text="Auto-start live translation on open",
            variable=self.auto_start_live,
            bootstyle="round-toggle",
        ).pack(anchor=W, pady=(6, 0))

        trans = tbs.Labelframe(right_col, text="Live translation", padding=(12, 10))
        trans.pack(fill=BOTH, expand=True, pady=(0, 8))

        if TranscriptText is not None:
            self.transcript = TranscriptText(
                trans,
                height=6,
                font=("Consolas", 11),
                wrap=tk.WORD,
                bootstyle=SECONDARY,
            )
            self.transcript.pack(fill=BOTH, expand=True, pady=(0, 8))
        else:
            row = tbs.Frame(trans)
            row.pack(fill=BOTH, expand=True, pady=(0, 8))
            sb = tbs.Scrollbar(row, bootstyle=ROUND)
            self.transcript = tk.Text(
                row,
                height=6,
                font=("Consolas", 11),
                wrap=tk.WORD,
                yscrollcommand=sb.set,
                bg="#2b2b2b",
                fg="#e0e0e0",
                insertbackground="#e0e0e0",
                highlightthickness=0,
                borderwidth=0,
            )
            sb.config(command=self.transcript.yview)
            self.transcript.pack(side=LEFT, fill=BOTH, expand=True)
            sb.pack(side=RIGHT, fill=Y)

        self.transcript_body = self.transcript.text if hasattr(self.transcript, "text") else self.transcript

        self.transcript_body.insert("1.0", "(Completed signs appear here as a sentence.)")
        self.transcript_body.configure(state="disabled")

        trans_btns = tbs.Frame(trans)
        trans_btns.pack(fill=X)
        tbs.Button(
            trans_btns,
            text="Clear line",
            bootstyle=SECONDARY,
            width=12,
            command=self.controller.clear_translation,
        ).pack(side=LEFT, padx=(0, 8))
        tbs.Button(
            trans_btns,
            text="Read line aloud",
            bootstyle=INFO,
            width=14,
            command=self.controller.speak_translation_line,
        ).pack(side=LEFT, padx=2)

        actions = tbs.Labelframe(right_col, text="Actions", padding=(12, 10))
        actions.pack(fill=X, pady=(0, 8))

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

        row2 = tbs.Frame(actions)
        row2.pack(fill=X)
        tbs.Button(
            row2,
            text="Start live",
            bootstyle=PRIMARY,
            width=14,
            command=self.controller.use_ai,
        ).pack(side=LEFT, padx=(0, 6))
        tbs.Button(
            row2,
            text="Stop AI",
            bootstyle=DANGER,
            width=14,
            command=self.controller.stop_ai,
        ).pack(side=LEFT, padx=6)

        anim = tbs.Labelframe(right_col, text="Text → sign animation", padding=(12, 10))
        anim.pack(fill=X, pady=(0, 4))

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

    def append_translation(self, token: str) -> None:
        t = (token or "").strip().upper()
        if not t or t in IGNORE_PREDICTIONS:
            return
        self.transcript_body.configure(state="normal")
        cur = self.transcript_body.get("1.0", "end").strip()
        placeholder = "(Completed signs appear here as a sentence.)"
        if cur == placeholder or not cur:
            self.transcript_body.delete("1.0", "end")
            self.transcript_body.insert("1.0", t)
        else:
            self.transcript_body.insert("end", " " + t)
        self.transcript_body.configure(state="disabled")
        self.transcript_body.see("end")

    def clear_translation_text(self) -> None:
        self.transcript_body.configure(state="normal")
        self.transcript_body.delete("1.0", "end")
        self.transcript_body.insert("1.0", "(Completed signs appear here as a sentence.)")
        self.transcript_body.configure(state="disabled")

    def get_translation_text(self) -> str:
        raw = self.transcript_body.get("1.0", "end").strip()
        ph = "(Completed signs appear here as a sentence.)"
        if raw == ph:
            return ""
        return raw

    def run(self) -> None:
        self.root.mainloop()
