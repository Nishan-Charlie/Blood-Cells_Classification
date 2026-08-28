"""Build the standalone project page by inlining the figures as data URIs.

The published page has to be self-contained: the artifact host blocks external
image sources, so every figure is downsampled and base64-embedded here rather
than linked. Photographic panels become JPEG, plots stay PNG (flat colour and
thin axis rules ring badly under JPEG).

    python scripts/build_project_page.py
    -> docs/index.html
"""
import base64
import io
import re
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts"
TEMPLATE = REPO / "scripts" / "project_page_template.html"

# Two outputs from one template, because the two hosts want different things.
#   docs/index.html    a complete HTML document, for GitHub Pages. Needs its own
#                      <html>/<head>, charset and viewport or it is not
#                      responsive on mobile.
#   docs/artifact.html the bare fragment, for the Claude artifact host, which
#                      supplies its own <!doctype>/<head>/<body> wrapper at
#                      publish time. Adding one here would nest them.
OUT_PAGE = REPO / "docs" / "index.html"
OUT_FRAGMENT = REPO / "docs" / "artifact.html"

DESCRIPTION = ("Lineage-aware hierarchical deep learning for imbalanced "
               "classification of peripheral blood cells: method, results and "
               "data for an MSc dissertation on the MLL23 corpus.")

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

    OUT_PAGE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FRAGMENT.write_text(html, encoding="utf-8")

    # Lift <title> and the font <link> out of the fragment into a real <head>.
    title_m = re.search(r"<title>(.*?)</title>", html, re.S)
    title = title_m.group(1).strip() if title_m else "Project page"
    fonts = re.findall(r'<link rel="stylesheet"[^>]*>', html)
    body = re.sub(r"<title>.*?</title>\s*", "", html, count=1, flags=re.S)
    for f in fonts:
        body = body.replace(f, "", 1)

    head = "\n".join([
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{title}</title>",
        f'<meta name="description" content="{DESCRIPTION}">',
        f'<meta property="og:title" content="{title}">',
        f'<meta property="og:description" content="{DESCRIPTION}">',
        '<meta property="og:type" content="article">',
        '<meta name="twitter:card" content="summary_large_image">',
        *fonts,
        "</head>",
        "<body>",
    ])
    OUT_PAGE.write_text(head + body.lstrip() + "\n</body>\n</html>\n",
                        encoding="utf-8")

    for p in (OUT_PAGE, OUT_FRAGMENT):
        print(f"  wrote {p.relative_to(REPO)}  "
              f"({p.stat().st_size/1024/1024:.2f} MB)")


if __name__ == "__main__":
    main()
