#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 14:31:14 2025

@author: jason
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
from PIL import Image

# === Paths & filenames ===
GALAXY_DIR = Path("/Users/jason/Desktop/Unique COSMOS-Web Sources/")
PSF_DIR    = Path("/Users/jason/Desktop/Starbirth - Oxford Toomre Q Project/Project/")  # change if your psf_image_*.png are elsewhere
OUT_DIR    = GALAXY_DIR / "galaxy_psf_stacks"

# Exact galaxy filenames (must exist in GALAXY_DIR)
GALAXY_FILES = [
    "Unique COSMOS-Web Source 3178.png",
    "Unique COSMOS-Web Source 4951.png",
    "Unique COSMOS-Web Source 3983.png",
    "Unique COSMOS-Web Source 4605.png",
    "Unique COSMOS-Web Source 5367.png",
]

# PSF sequence
N_PSFS = 49  # psf_image_1.png ... psf_image_49.png

# Blend control
PSF_STRENGTH = 1.0         # increase if PSFs look faint, decrease if too bright
NORMALIZE_PSF_PEAK = False  # makes PSF brightness comparable across the set


def load_image_rgb(path):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr, im.size  # (W, H)


def save_image_rgb(arr, path):
    arr = np.clip(arr, 0.0, 1.0)
    im = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def center_overlay_bounds(bg_h, bg_w, fg_h, fg_w):
    cy_bg, cx_bg = bg_h // 2, bg_w // 2
    cy_fg, cx_fg = fg_h // 2, fg_w // 2

    ys_bg = cy_bg - cy_fg
    xs_bg = cx_bg - cx_fg
    ye_bg = ys_bg + fg_h
    xe_bg = xs_bg + fg_w

    ys_bg_clip = max(0, ys_bg)
    xs_bg_clip = max(0, xs_bg)
    ye_bg_clip = min(bg_h, ye_bg)
    xe_bg_clip = min(bg_w, xe_bg)

    ys_fg_clip = ys_bg_clip - ys_bg
    xs_fg_clip = xs_bg_clip - xs_bg
    ye_fg_clip = ys_fg_clip + (ye_bg_clip - ys_bg_clip)
    xe_fg_clip = xs_fg_clip + (xe_bg_clip - xs_bg_clip)

    return (ys_bg_clip, ye_bg_clip, xs_bg_clip, xe_bg_clip), (ys_fg_clip, ye_fg_clip, xs_fg_clip, xe_fg_clip)


def normalize_peak(psf_rgb):
    peak = float(np.max(psf_rgb))
    if peak > 0:
        return psf_rgb / peak
    return psf_rgb


def blend_add(bg_roi, fg_roi, strength):
    return np.clip(bg_roi + fg_roi * strength, 0.0, 1.0)


def main():
    # Check galaxies exist
    galaxy_paths = []
    for name in GALAXY_FILES:
        p = GALAXY_DIR / name
        if not p.exists():
            raise FileNotFoundError(f"Galaxy not found: {p}")
        galaxy_paths.append(p)

    # Process each galaxy against all PSFs
    for gpath in galaxy_paths:
        galaxy_rgb, (gW, gH) = load_image_rgb(gpath)
        g_base = gpath.stem
        out_subdir = OUT_DIR / g_base

        for i in range(1, N_PSFS + 1):
            psf_path = PSF_DIR / f"psf_image_{i}.png"
            if not psf_path.exists():
                raise FileNotFoundError(f"Missing PSF: {psf_path}")

            psf_rgb, (pW, pH) = load_image_rgb(psf_path)
            if NORMALIZE_PSF_PEAK:
                psf_rgb = normalize_peak(psf_rgb)

            out = galaxy_rgb.copy()

            (ysb, yeb, xsb, xeb), (ysf, yef, xsf, xef) = center_overlay_bounds(
                bg_h=gH, bg_w=gW, fg_h=pH, fg_w=pW
            )

            bg_roi = out[ysb:yeb, xsb:xeb, :]
            fg_roi = psf_rgb[ysf:yef, xsf:xef, :]

            out[ysb:yeb, xsb:xeb, :] = blend_add(bg_roi, fg_roi, PSF_STRENGTH)

            out_name = f"{g_base}__psf{i:02d}.png"
            save_image_rgb(out, out_subdir / out_name)
            print(f"Saved: {out_subdir/out_name}")

    print("\nAll done.")
    print(f"Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
