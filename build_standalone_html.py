"""Render BLOG_POST.md to a single self-contained HTML file.

All ![alt](path) image references are inlined as base64 data URIs so the
resulting HTML opens correctly without the field_plots/ folder.

Usage:
    python build_standalone_html.py
        --in  BLOG_POST.md
        --out BLOG_POST.html
"""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

import markdown

IMG_MD = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
MIMES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".svg": "image/svg+xml", ".webp": "image/webp",
}

CSS = """
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Helvetica Neue", Arial, sans-serif;
  max-width: 820px;
  margin: 40px auto;
  padding: 0 24px;
  color: #1a1a1a;
  line-height: 1.6;
}
h1 { font-size: 1.9em; line-height: 1.25; margin-top: 0.8em; }
h2 { font-size: 1.4em; margin-top: 1.6em; border-bottom: 1px solid #eee;
     padding-bottom: 0.2em; }
p, li { font-size: 1.05em; }
img { display: block; margin: 1.2em auto; max-width: 100%; height: auto;
      border-radius: 4px; box-shadow: 0 1px 6px rgba(0,0,0,0.08); }
figcaption, .caption { display: block; text-align: center; font-size: 0.9em;
                       color: #555; margin: -0.6em auto 1.4em; max-width: 90%; }
table { border-collapse: collapse; margin: 1.2em auto; font-size: 0.95em; }
th, td { border: 1px solid #ddd; padding: 6px 12px; text-align: right; }
th { background: #f6f6f6; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px;
       font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; }
pre { background: #f4f4f4; padding: 10px; border-radius: 4px; overflow: auto; }
blockquote { color: #555; border-left: 4px solid #ddd; margin: 1em 0;
             padding-left: 1em; }
"""


def _to_data_uri(path: Path) -> str:
    ext = path.suffix.lower()
    mime = MIMES.get(ext, "application/octet-stream")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _inline_images(md_text: str, root: Path) -> tuple[str, list[tuple[str, str]]]:
    """Replace ![alt](path) with ![alt](data:...) for local image paths.

    Returns the rewritten markdown and a list of (alt, original_path) for
    figcaption emission after markdown is converted.
    """
    captions: list[tuple[str, str]] = []

    def _repl(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        p = (root / src).resolve()
        if not p.is_file():
            print(f"WARN: missing image {src}")
            return m.group(0)
        uri = _to_data_uri(p)
        captions.append((alt, src))
        return f"![{alt}]({uri})"

    return IMG_MD.sub(_repl, md_text), captions


def _wrap_imgs_with_figcaption(html: str) -> str:
    """For every <img alt="...">, wrap it in a <figure> with a figcaption."""
    pattern = re.compile(
        r'<img\s+(?P<attrs>[^>]*?)alt="(?P<alt>[^"]*)"(?P<rest>[^>]*)/?>',
        re.DOTALL,
    )
    def _repl(m: re.Match) -> str:
        alt = m.group("alt").strip()
        if not alt:
            return m.group(0)
        return (f'<figure>{m.group(0)}'
                f'<figcaption>{alt}</figcaption></figure>')
    return pattern.sub(_repl, html)


def build(md_path: Path, out_path: Path):
    text = md_path.read_text(encoding="utf-8")
    rewritten, _ = _inline_images(text, md_path.parent)
    body = markdown.markdown(
        rewritten,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    body = _wrap_imgs_with_figcaption(body)
    title = "Designing Ten Modulators Overnight"
    html = (
        "<!doctype html>\n<html lang='en'><head>"
        f"<meta charset='utf-8'><title>{title}</title>"
        f"<style>{CSS}</style></head><body>\n{body}\n</body></html>"
    )
    out_path.write_bytes(html.encode("utf-8"))
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path}  ({size_mb:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="md", type=Path, default=Path("BLOG_POST.md"))
    p.add_argument("--out", type=Path, default=Path("BLOG_POST.html"))
    args = p.parse_args()
    build(args.md, args.out)


if __name__ == "__main__":
    main()
