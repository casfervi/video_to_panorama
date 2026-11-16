#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Long image from a video that moves in ONE direction, using a strip mosaic (pushbroom).

No feature matching and no homographies. For every frame, a narrow strip is cut from the
centre of the frame and the strips are joined side by side. The width of each strip is the
distance the scene moved in the video at that moment, measured with phase correlation, so
speed changes are handled automatically. Every pixel comes straight from the video at full
resolution.

Works best when the camera moves in a straight line, at a roughly constant distance from a
mostly flat scene. It does not correct parallax, shake or rolling shutter.

Examples
--------
    python video_strip_mosaic.py video.mp4
    python video_strip_mosaic.py video.mp4 --axis y --start-frame 120 --end-frame 900
    python video_strip_mosaic.py video.mp4 --step 2 --crop-ratio 0.8 -o long.png
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

JPEG_MAX_SIDE = 65500   # JPEG cannot store images larger than this on either side


# --------------------------------------------------------------------------
# Video reading
# --------------------------------------------------------------------------
def iter_frames(path, start, end, step, crop_ratio, rotate):
    """Yield frames start, start+step, ... (before `end`), cropped to the top `crop_ratio`."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    idx = 0
    while True:
        if not cap.grab():          # grab() skips decoding of frames we don't use
            break
        if end is not None and idx >= end:
            break
        if idx >= start and (idx - start) % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if crop_ratio < 1.0:
                frame = frame[: int(frame.shape[0] * crop_ratio)]
            if rotate:              # vertical motion -> work as if it were horizontal
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            yield frame
        idx += 1
    cap.release()


# --------------------------------------------------------------------------
# Pass 1: motion between consecutive frames
# --------------------------------------------------------------------------
def estimate_shifts(frames, analysis_width):
    """Return array (N-1, 3): dx, dy (in full-resolution pixels) and correlation response."""
    out, prev, window, n = [], None, None, 0
    for f in frames:
        n += 1
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        s = 1.0
        if analysis_width and g.shape[1] > analysis_width:
            s = analysis_width / g.shape[1]
            g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        g = g.astype(np.float32)
        if prev is not None:
            if window is None:
                window = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
            (dx, dy), resp = cv2.phaseCorrelate(prev, g, window)
            out.append((dx / s, dy / s, resp))
        prev = g
        if n % 100 == 0:
            print(f"  analysed {n} frames")
    if n < 2:
        raise RuntimeError("Need at least 2 frames (check --start-frame / --end-frame / --step).")
    return np.array(out, dtype=np.float64), n


def clean_shifts(dx, resp, frame_w, min_response, smooth):
    """Drop unreliable measurements, fill the gaps by interpolation, median-smooth."""
    dx = dx.copy()
    bad = (resp < min_response) | (np.abs(dx) > 0.9 * frame_w)
    if bad.mean() > 0.5:
        raise RuntimeError(f"{bad.mean():.0%} of the shift measurements are unreliable. "
                           "Lower --min-response, use a smaller --step, or --crop-ratio to drop "
                           "moving/low-texture areas.")
    idx = np.arange(len(dx))
    dx[bad] = np.interp(idx[bad], idx[~bad], dx[~bad])
    if smooth > 1:
        k = smooth | 1                                   # odd window
        pad = np.pad(dx, k // 2, mode="edge")
        dx = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
    print(f"Unreliable measurements replaced: {int(bad.sum())}/{len(bad)}")
    return dx


# --------------------------------------------------------------------------
# Pass 2: cut strips and join them
# --------------------------------------------------------------------------
def build_mosaic(frames, advance, center_frac):
    strips, cum, emitted = [], 0.0, 0
    for f, adv in zip(frames, advance):
        W = f.shape[1]
        cum += adv
        w = int(round(cum)) - emitted     # cumulative rounding: no drift in the total length
        if w <= 0:
            continue
        if w > W:
            raise RuntimeError(f"The scene moves {w}px between frames, more than the frame width "
                               f"({W}px). Use a smaller --step.")
        x0 = int(round(center_frac * W - w / 2))
        x0 = max(0, min(W - w, x0))
        strips.append(f[:, x0:x0 + w])
        emitted += w
    return strips


def parse_args():
    p = argparse.ArgumentParser(description="Strip mosaic: long image from a one-direction video.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path, help="Input video")
    p.add_argument("--axis", choices=["x", "y"], default="x",
                   help="Direction of the movement: x = horizontal, y = vertical (default: x)")
    p.add_argument("--start-frame", type=int, default=0, help="First frame to use (default: 0)")
    p.add_argument("--end-frame", type=int, default=None, help="Stop before this frame (default: end of video)")
    p.add_argument("--step", type=int, default=1,
                   help="Use one frame every N (default: 1). Raise it if the video is very slow/long.")
    p.add_argument("--crop-ratio", type=float, default=1.0,
                   help="Keep only the top fraction of each frame, 1.0 = no crop (default: 1.0)")
    p.add_argument("--strip-center", type=float, default=0.5,
                   help="Where in the frame to cut the strip, 0 = edge, 0.5 = centre, 1 = other edge (default: 0.5)")
    p.add_argument("--analysis-width", type=int, default=640,
                   help="Width used to measure motion, 0 = full resolution (default: 640)")
    p.add_argument("--smooth", type=int, default=5, help="Median filter size for the speed, 1 = off (default: 5)")
    p.add_argument("--min-response", type=float, default=0.05,
                   help="Minimum phase-correlation confidence to trust a measurement (default: 0.05)")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="Output image (default: output/strip_<datetime>.jpg)")
    a = p.parse_args()

    if not a.video.is_file():
        p.error(f"Video not found: {a.video}")
    if a.step < 1:
        p.error("--step must be >= 1")
    if not 0 < a.crop_ratio <= 1:
        p.error("--crop-ratio must be in (0, 1]")
    if not 0 <= a.strip_center <= 1:
        p.error("--strip-center must be in [0, 1]")
    return a


def main():
    a = parse_args()
    rotate = a.axis == "y"
    make = lambda: iter_frames(a.video, a.start_frame, a.end_frame, a.step, a.crop_ratio, rotate)

    try:
        print("Pass 1/2: measuring motion")
        shifts, n = estimate_shifts(make(), a.analysis_width)
        W = next(make()).shape[1]

        dx = clean_shifts(shifts[:, 0], shifts[:, 2], W, a.min_response, a.smooth)
        direction = np.sign(np.median(dx))
        if direction == 0:
            raise RuntimeError("No motion detected between frames.")
        moved = np.clip(dx * direction, 0, None)        # how far the scene moved, always >= 0
        # frame i represents the interval around it: average of the shift before and after it
        advance = np.empty(n)
        advance[0], advance[-1] = moved[0], moved[-1]
        advance[1:-1] = 0.5 * (moved[:-1] + moved[1:])
        print(f"{n} frames | mean motion {moved.mean():.2f}px/frame "
              f"(min {moved.min():.2f}, max {moved.max():.2f}) | expected length ~{advance.sum():.0f}px")

        print("Pass 2/2: cutting strips")
        strips = build_mosaic(make(), advance, a.strip_center)
        # scene moving right (dx > 0) means later frames show what lies to the LEFT
        if direction > 0:
            strips.reverse()
        mosaic = np.hstack(strips)
        if rotate:
            mosaic = cv2.rotate(mosaic, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    out = a.output
    if out is None:
        Path("output").mkdir(exist_ok=True)
        out = Path("output") / f"strip_{datetime.now():%Y%m%d_%H%M%S}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    if max(mosaic.shape[:2]) > JPEG_MAX_SIDE and out.suffix.lower() in (".jpg", ".jpeg"):
        out = out.with_suffix(".png")
        print(f"Image is larger than the JPEG limit ({JPEG_MAX_SIDE}px): saving as PNG instead.")

    if not cv2.imwrite(str(out), mosaic):
        print(f"Could not write {out}", file=sys.stderr)
        sys.exit(1)
    print(f"\nSaved: {out}  ({mosaic.shape[1]} x {mosaic.shape[0]}, {len(strips)} strips)")


if __name__ == "__main__":
    main()
