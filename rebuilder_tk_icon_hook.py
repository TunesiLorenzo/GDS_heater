"""PyInstaller runtime hook: apply the rebuilder's icon to Tkinter windows."""
from pathlib import Path
import sys


def _install_tk_icon():
    if not getattr(sys, 'frozen', False):
        return
    assets = Path(getattr(sys, '_MEIPASS', '')) / '_rebuilder_assets'
    icons = list(assets.glob('*.ico'))
    if len(icons) != 1:
        return
    try:
        import tkinter as tk
    except ImportError:
        return  # Not a Tkinter application.

    original_init = tk.Tk.__init__
    icon_path = str(icons[0])

    def tk_init_with_icon(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            # -default also applies to Toplevel windows unless they override it.
            self.iconbitmap(default=icon_path)
        except tk.TclError:
            pass

    tk.Tk.__init__ = tk_init_with_icon


_install_tk_icon()
