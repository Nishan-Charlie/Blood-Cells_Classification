"""Build the standalone project page by inlining the figures as data URIs.

The published page has to be self-contained: the artifact host blocks external
image sources, so every figure is downsampled and base64-embedded here rather
than linked. Photographic panels become JPEG, plots stay PNG (flat colour and
thin axis rules ring badly under JPEG).

    python scripts/build_project_page.py
    -> docs/project_page.html
"""
import base64
import io
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts"
TEMPLATE = REPO / "scripts" / "project_page_template.html"
OUT = REPO / "docs" / "project_page.html"

# key -> (source file, max width in px, encoding)
FIGURES = {
    "cells":        ("class_examples.png",         1500, "jpeg"),
    "dist":         ("class_distribution.png",     1400, "png"),
    "architecture": ("fig_architecture.png",       1200, "png"),
    "arms":         ("fig_arm_comparison.png",     1250, "png"),
    "perclass":     ("fig_per_class_dumbbell.png", 1250, "png"),
    "stage":        ("fig_stage_distance.png",     1150, "png"),
    "gradcam":      ("gradcam_fine.png",           1300, "jpeg"),
}


def encode(path: Path, maxw: int, mode: str) -> str:
    im = Image.open(path)
    if im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    if mode == "jpeg":
        im.convert("RGB").save(buf, "JPEG", quality=82, optimize=True,
                               progressive=True)
        mime = "image/jpeg"
    else:
        im.convert("P", palette=Image.ADAPTIVE, colors=192).save(
            buf, "PNG", optimize=True)
        mime = "image/png"
    raw = buf.getvalue()
    print(f"  {path.name:32s} {path.stat().st_size/1024:7.0f} KB -> "
          f"{len(raw)/1024:6.0f} KB")
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def main() -> None:
    html = TEMPLATE.read_text(encoding="utf-8")
    for key, (fname, maxw, mode) in FIGURES.items():
        src = ART / fname
        if not src.exists():
            raise FileNotFoundError(
                f"{src} is missing. Run scripts/build_dissertation/make_figures.py "
                "to regenerate the result figures.")
        html = html.replace("{{" + key + "}}", encode(src, maxw, mode))

    if "{{" in html:
        leftover = html[html.index("{{"):html.index("{{") + 40]
        raise RuntimeError(f"unsubstituted placeholder near: {leftover!r}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"\nwrote {OUT}  ({OUT.stat().st_size/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
