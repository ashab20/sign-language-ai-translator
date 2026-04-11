import tkinter as tk

from controller.app_controller import AppController
from view.main_view import MainView

if __name__ == "__main__":
    controller = AppController()
    view = MainView(controller)
    controller.view = view

    def _start_live_translation_if_enabled() -> None:
        try:
            if not view.root.winfo_exists():
                return
            if view.auto_start_live.get():
                controller.use_ai()
        except tk.TclError:
            pass

    if hasattr(view, "auto_start_live"):
        view.root.after(250, _start_live_translation_if_enabled)
    view.run()