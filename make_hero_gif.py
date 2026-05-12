"""Combine the 3D modulator image + doping GIF + CPS-evolution GIF into one
animated hero figure for the blog post.

Layout: 2-row stack
  Row 1: 3D modulator (static), full width
  Row 2: Step-1 doping (animated) | Step-2 CPS (animated)

The two GIFs have different frame counts (10 and 20). We use LCM(10, 20) = 20
composite frames: doping advances every 2 frames, CPS advances every 1 frame.

Output: field_plots/hero_animation.gif
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageSequence

ROOT = Path("field_plots")
OUT = ROOT / "hero_animation.gif"

TOTAL_WIDTH = 1400            # px — overall canvas width
TOP_HEIGHT = 460              # px — height of the 3D row
BOTTOM_HEIGHT = 520           # px — height of each bottom GIF panel
GAP = 18                      # px between panels
TITLE_HEIGHT = 32             # px reserved at the top for panel titles
BG_COLOR = (255, 255, 255)
TITLE_COLOR = (40, 40, 40)
FRAME_DURATION_MS = 700       # per composite frame


def _load_gif_frames(path: Path) -> list[Image.Image]:
    img = Image.open(path)
    frames = []
    for f in ImageSequence.Iterator(img):
        # Convert to RGBA so we don't lose transparency, then composite onto white
        rgba = f.convert("RGBA")
        bg = Image.new("RGB", rgba.size, BG_COLOR)
        bg.paste(rgba, mask=rgba.split()[-1])
        frames.append(bg.copy())
    return frames


def _resize_to_height(img: Image.Image, h: int) -> Image.Image:
    new_w = int(round(img.width * h / img.height))
    return img.resize((new_w, h), Image.LANCZOS)


def _resize_to_width(img: Image.Image, w: int) -> Image.Image:
    new_h = int(round(img.height * w / img.width))
    return img.resize((w, new_h), Image.LANCZOS)


def _fit_in_box(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Scale `img` to fit inside (max_w × max_h) preserving aspect ratio."""
    scale = min(max_w / img.width, max_h / img.height)
    new_w = int(round(img.width * scale))
    new_h = int(round(img.height * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _font(size: int) -> ImageFont.FreeTypeFont:
    # Try a few common system fonts; fall back to default.
    for name in ["arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf",
                 "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _label_panel(img: Image.Image, text: str) -> Image.Image:
    """Return a new image with a centred title bar above the panel."""
    out = Image.new("RGB", (img.width, img.height + TITLE_HEIGHT), BG_COLOR)
    out.paste(img, (0, TITLE_HEIGHT))
    draw = ImageDraw.Draw(out)
    font = _font(15)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.width - tw) // 2, 4), text, fill=TITLE_COLOR, font=font)
    return out


def main():
    img3d = Image.open(ROOT / "modulator_3d.png").convert("RGB")
    doping_frames = _load_gif_frames(ROOT / "doping_sweep.gif")
    cps_frames = _load_gif_frames(ROOT / "cps_evolution.gif")
    n_doping = len(doping_frames)
    n_cps = len(cps_frames)
    print(f"loaded: 3d=1, doping={n_doping}, cps={n_cps}")

    # Top row: 3D image scaled to the full canvas width.
    img3d_scaled = _resize_to_width(img3d, TOTAL_WIDTH)
    if img3d_scaled.height > TOP_HEIGHT:
        img3d_scaled = _fit_in_box(img3d, TOTAL_WIDTH, TOP_HEIGHT)

    # Bottom row: each GIF gets half the canvas (minus the central gap).
    bottom_panel_w = (TOTAL_WIDTH - GAP) // 2
    doping_frames = [_fit_in_box(f, bottom_panel_w, BOTTOM_HEIGHT)
                     for f in doping_frames]
    cps_frames = [_fit_in_box(f, bottom_panel_w, BOTTOM_HEIGHT)
                  for f in cps_frames]

    # Add titles to each panel
    img3d_t = _label_panel(img3d_scaled,
                            "3D view: segmented CPS over the silicon MZM")
    doping_titles = [_label_panel(f, "Step 1 — doping sweep (mult 0.2 → 20)")
                     for f in doping_frames]
    cps_titles = [_label_panel(f,
                                "Step 2 — electrode BO trajectory (C = 4 pF/cm)")
                  for f in cps_frames]

    # Final canvas size
    bottom_h = max(doping_titles[0].height, cps_titles[0].height)
    h_total = img3d_t.height + GAP + bottom_h

    lcm_frames = (n_doping * n_cps) // gcd(n_doping, n_cps)

    composite_frames = []
    for k in range(lcm_frames):
        d_idx = (k * n_doping) // lcm_frames
        c_idx = (k * n_cps) // lcm_frames
        canvas = Image.new("RGB", (TOTAL_WIDTH, h_total), BG_COLOR)
        # Top row, centred
        canvas.paste(img3d_t, ((TOTAL_WIDTH - img3d_t.width) // 2, 0))
        # Bottom row
        y0 = img3d_t.height + GAP
        dp = doping_titles[d_idx]
        cp = cps_titles[c_idx]
        # Centre each panel in its half-canvas
        x_dp = (bottom_panel_w - dp.width) // 2
        x_cp = bottom_panel_w + GAP + (bottom_panel_w - cp.width) // 2
        canvas.paste(dp, (x_dp, y0))
        canvas.paste(cp, (x_cp, y0))
        composite_frames.append(canvas)

    composite_frames[0].save(
        OUT,
        save_all=True,
        append_images=composite_frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_mb = OUT.stat().st_size / 1e6
    print(f"Wrote {OUT}  ({lcm_frames} frames, {size_mb:.2f} MB,"
          f" {TOTAL_WIDTH}x{h_total} px)")


if __name__ == "__main__":
    main()
