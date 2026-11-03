#!/usr/bin/env python3
"""Generate one animated GIF per figure category from docs/figures/*.png."""
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("pip install pillow")

FIG_DIR = Path("docs/figures")
OUT_DIR = FIG_DIR / "gifs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = {
    "spam_models": ["spam_class_distribution", "spam_pr_curve", "spam_pr_auc_by_model",
                    "spam_roc_curve", "spam_threshold_curve", "spam_roc_vs_pr",
                    "spam_threshold_sensitivity", "spam_holm_significance"],
    "fraud_models": ["fraud_class_distribution", "fraud_pr_curve", "fraud_pr_auc_by_model",
                     "fraud_roc_curve", "fraud_threshold_curve", "fraud_roc_vs_pr",
                     "fraud_threshold_sensitivity", "fraud_holm_significance"],
    "framework_artifacts": ["framework_pipeline_diagram", "framework_layer_order",
                            "framework_mcnemar_decision_tree", "framework_threshold_cost_curve"],
    "readme_hero": ["readme_hero_results", "readme_results_table", "readme_personas_triptych"],
}
FRAME_MS = 1400
SCALE = 0.85


def load_font(size):
    for name in ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def annotate(img, label):
    img = img.convert("RGB")
    w, h = img.size
    new_size = (int(w * SCALE), int(h * SCALE))
    img = img.resize(new_size, Image.LANCZOS)
    band = 34
    canvas = Image.new("RGB", (new_size[0], new_size[1] + band), (255, 255, 255))
    canvas.paste(img, (0, band))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), label, fill=(0, 0, 0), font=load_font(20))
    return canvas


def collect(names):
    out = []
    for name in names:
        f = FIG_DIR / f"{name}.png"
        if f.exists():
            out.append(f)
    return out


def build_gif(name, files):
    if len(files) < 2:
        print(f"[skip] {name}: {len(files)} file(s), need >=2")
        return None
    frames = [annotate(Image.open(f), f.name) for f in files]
    w = max(f.size[0] for f in frames)
    h = max(f.size[1] for f in frames)
    frames = [f.resize((w, h), Image.LANCZOS) if f.size != (w, h) else f for f in frames]
    out = OUT_DIR / f"{name}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=FRAME_MS, loop=0, optimize=True)
    print(f"[ok]   {name}.gif  ({len(frames)} frames, {out.stat().st_size // 1024} KiB)")
    return out


def main():
    if not FIG_DIR.exists():
        sys.exit(f"missing {FIG_DIR}")
    built = 0
    for name, pattern_names in CATEGORIES.items():
        files = collect(pattern_names)
        if build_gif(name, files):
            built += 1
    if built == 0:
        sys.exit("no GIFs built — check CATEGORIES glob patterns against docs/figures/")
    print(f"\nbuilt {built} GIF(s) in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
