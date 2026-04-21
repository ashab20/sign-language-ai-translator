"""Resolve a university logo file from assets (raster or SVG)."""

import os
import shutil
import subprocess
import sys

from config import ASSETS_DIR, DATA_DIR, UNIVERSITY_LOGO_RASTER_FILES, UNIVERSITY_LOGO_SVG_FILES


def _svg_via_cairosvg(svg_path: str, out_png: str) -> bool:
    try:
        import cairosvg
        from io import BytesIO

        png_bytes = cairosvg.svg2png(url=svg_path, output_width=300, output_height=360)
        with open(out_png, "wb") as f:
            f.write(png_bytes)
        return True
    except Exception:
        return False


def _svg_via_qlmanage(svg_path: str, out_png: str) -> bool:
    if sys.platform != "darwin":
        return False
    if not shutil.which("qlmanage"):
        return False
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        subprocess.run(
            ["qlmanage", "-t", "-s", "300", "-o", DATA_DIR, os.path.abspath(svg_path)],
            check=False,
            capture_output=True,
            timeout=15,
        )
        thumb = os.path.join(DATA_DIR, os.path.basename(svg_path) + ".png")
        if os.path.isfile(thumb):
            shutil.copy2(thumb, out_png)
            return True
    except (subprocess.TimeoutExpired, OSError):
        pass
    return False


def resolve_university_logo_png() -> str | None:
    """Return path to a PNG (or other PIL-readable) file for the About tab, or None.

    Order: raster files in assets/, then SVG (cairosvg if available, else qlmanage on macOS).
    Cached raster is written to data/university_logo_cache.png when converting from SVG.
    """
    for name in UNIVERSITY_LOGO_RASTER_FILES:
        path = os.path.join(ASSETS_DIR, name)
        if os.path.isfile(path):
            return path

    cache_png = os.path.join(DATA_DIR, "university_logo_cache.png")
    for name in UNIVERSITY_LOGO_SVG_FILES:
        svg_path = os.path.join(ASSETS_DIR, name)
        if not os.path.isfile(svg_path):
            continue

        if os.path.isfile(cache_png):
            try:
                if os.path.getmtime(cache_png) >= os.path.getmtime(svg_path):
                    return cache_png
            except OSError:
                pass

        os.makedirs(DATA_DIR, exist_ok=True)
        if _svg_via_cairosvg(svg_path, cache_png):
            return cache_png
        if _svg_via_qlmanage(svg_path, cache_png):
            return cache_png

    return None
