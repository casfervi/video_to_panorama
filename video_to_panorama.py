#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video -> panorama using SIFT + homographies + feathered blending.

Usage
-----
    python video_to_panorama.py "C:\\path\\video.mp4" --step 30
    python video_to_panorama.py video.mp4 -s 15 -o pano.jpg --crop-ratio 0.7
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import os
from datetime import datetime

# Safety limits
MAX_CANVAS_PIXELS = 80_000_000   # panorama is downscaled if it would exceed this
MIN_INLIERS = 15                 # minimum RANSAC inliers to accept a pair
MIN_INLIER_RATIO = 0.25          # minimum inliers / good matches
MAX_PERSPECTIVE = 5e-4           # reject wild perspective terms in H
SCALE_RANGE = (0.5, 2.0)         # allowed area scale between consecutive frames


# --------------------------------------------------------------------------
# Frame extraction
# --------------------------------------------------------------------------
def extract_frames(video_path, step=30, crop_ratio=0.70, max_width=1600):
    """
    Read one frame every `step` frames.

    crop_ratio : keep only the TOP `crop_ratio` fraction of each frame (1.0 = no crop).
    max_width  : downscale frames wider than this (0 = disabled). Big speed/memory win.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    idx = 0
    while True:
        # grab() skips decoding/conversion work for frames we throw away
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break

            if crop_ratio < 1.0:
                frame = frame[: int(frame.shape[0] * crop_ratio), :]

            w = frame.shape[1]
            if max_width and w > max_width:
                s = max_width / w
                frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

            # .copy() so we don't keep the full uncropped frame alive in memory
            frames.append(np.ascontiguousarray(frame))
        idx += 1

    cap.release()
    print(f"Read {idx} video frames, kept {len(frames)} (step={step})")
    return frames


# --------------------------------------------------------------------------
# Features / matching
# --------------------------------------------------------------------------
def detect_features(frames, nfeatures=8000, contrast_threshold=0.01):
    """Compute SIFT once per frame (the original computed it twice)."""
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrast_threshold)
    feats = []
    for i, f in enumerate(frames):
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        kps, des = sift.detectAndCompute(gray, None)
        pts = np.float32([k.pt for k in kps]) if kps else np.empty((0, 2), np.float32)
        feats.append((pts, des))
        print(f"Frame {i}: {len(pts)} keypoints")
    return feats


def plausible(H):
    """Reject degenerate / absurd homographies."""
    if H is None or not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-12:
        return False
    H = H / H[2, 2]
    if abs(H[2, 0]) > MAX_PERSPECTIVE or abs(H[2, 1]) > MAX_PERSPECTIVE:
        return False
    det = np.linalg.det(H[:2, :2])
    return SCALE_RANGE[0] < det < SCALE_RANGE[1]


def estimate_homography(fa, fb, matcher, ratio=0.75):
    """Homography mapping frame B coordinates -> frame A coordinates."""
    pts_a, des_a = fa
    pts_b, des_b = fb
    if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
        return None

    pairs = matcher.knnMatch(des_a, des_b, k=2)
    # knnMatch can return fewer than 2 neighbours -> the original `for m, n in` could crash
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]

    if len(good) < MIN_INLIERS:
        print(f"  insufficient matches ({len(good)})")
        return None

    src = pts_b[[m.trainIdx for m in good]].reshape(-1, 1, 2)
    dst = pts_a[[m.queryIdx for m in good]].reshape(-1, 1, 2)

    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H, mask = cv2.findHomography(src, dst, method, 3.0)
    if H is None or mask is None:
        print("  homography failed")
        return None

    inliers = int(mask.sum())
    ratio_in = inliers / len(good)
    print(f"  matches={len(good)} inliers={inliers} ({100 * ratio_in:.1f}%)")

    if inliers < MIN_INLIERS or ratio_in < MIN_INLIER_RATIO or not plausible(H):
        print("  rejected")
        return None
    return H / H[2, 2]


def build_transforms(frames, feats):
    """
    Chain homographies. Frames that fail to match are DROPPED (the original duplicated
    the previous transform, stacking a misaligned frame on top of the previous one).
    A dropped frame doesn't become the reference: the next one is matched against the
    last good frame instead.
    """
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

    kept = [0]
    transforms = [np.eye(3)]
    ref = 0

    for i in range(1, len(frames)):
        print(f"Matching frame {ref} -> {i}")
        H = estimate_homography(feats[ref], feats[i], matcher)
        if H is None:
            continue
        T = transforms[-1] @ H
        transforms.append(T / T[2, 2])
        kept.append(i)
        ref = i

    # Re-anchor on the middle frame: far less distortion/canvas blow-up than anchoring
    # on the first frame, since errors accumulate in both directions instead of one.
    mid = len(transforms) // 2
    inv = np.linalg.inv(transforms[mid])
    transforms = [(inv @ T) / (inv @ T)[2, 2] for T in transforms]

    print(f"\nUsing {len(kept)}/{len(frames)} frames")
    return [frames[i] for i in kept], transforms


# --------------------------------------------------------------------------
# Warping / blending
# --------------------------------------------------------------------------
def frame_corners(frame, H):
    h, w = frame.shape[:2]
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(c, H).reshape(-1, 2)


def feather_weight(h, w):
    """Weight that is highest at the frame centre and ~0 at its borders."""
    m = np.zeros((h + 2, w + 2), np.uint8)
    m[1:-1, 1:-1] = 255
    return cv2.distanceTransform(m, cv2.DIST_L2, 3)[1:-1, 1:-1].astype(np.float32)


def warp_and_blend(frames, transforms):
    all_c = np.concatenate([frame_corners(f, H) for f, H in zip(frames, transforms)])
    xmin, ymin = np.floor(all_c.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(all_c.max(axis=0)).astype(int)

    width, height = int(xmax - xmin), int(ymax - ymin)
    scale = 1.0
    if width * height > MAX_CANVAS_PIXELS:
        scale = (MAX_CANVAS_PIXELS / (width * height)) ** 0.5
        width, height = int(np.ceil(width * scale)), int(np.ceil(height * scale))
        print(f"Canvas too large, downscaling by {scale:.2f}")
    print(f"Panorama size: {width} x {height}")

    # canvas = S @ Translate @ H
    base = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]]) @ \
           np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)

    acc = np.zeros((height, width, 3), np.float32)
    wsum = np.zeros((height, width), np.float32)

    for idx, (frame, H) in enumerate(zip(frames, transforms)):
        M = base @ H
        c = frame_corners(frame, M)
        x0 = max(int(np.floor(c[:, 0].min())), 0)
        y0 = max(int(np.floor(c[:, 1].min())), 0)
        x1 = min(int(np.ceil(c[:, 0].max())), width)
        y1 = min(int(np.ceil(c[:, 1].max())), height)
        if x1 <= x0 or y1 <= y0:
            continue

        print(f"Warping frame {idx + 1}/{len(frames)}")

        # Warp only into the frame's own bounding box (not the whole canvas)
        M_roi = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]]) @ M
        size = (x1 - x0, y1 - y0)

        warped = cv2.warpPerspective(frame, M_roi, size, flags=cv2.INTER_LINEAR)
        w = cv2.warpPerspective(feather_weight(*frame.shape[:2]), M_roi, size,
                                flags=cv2.INTER_LINEAR)

        acc[y0:y1, x0:x1] += warped.astype(np.float32) * w[:, :, None]
        wsum[y0:y1, x0:x1] += w

    valid = wsum > 1e-6
    pano = np.zeros_like(acc)
    pano[valid] = acc[valid] / wsum[valid][:, None]
    return np.clip(pano, 0, 255).astype(np.uint8), valid


def crop_to_valid(image, valid):
    """Crop using the real coverage mask (the old gray>0 test also dropped dark pixels)."""
    coords = cv2.findNonZero(valid.astype(np.uint8))
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y + h, x:x + w]


def create_panorama(frames, nfeatures=8000):
    feats = detect_features(frames, nfeatures=nfeatures)
    frames, transforms = build_transforms(frames, feats)
    if len(frames) < 2:
        raise RuntimeError("Could not align at least 2 frames. Try a smaller --step.")
    pano, valid = warp_and_blend(frames, transforms)
    return crop_to_valid(pano, valid)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Create a panorama from a video.")
    p.add_argument("video", help="Path to the input video")
    p.add_argument("-s", "--step", type=int, default=30,
                   help="Keep one frame every N frames (default: 30). "
                        "Smaller = more overlap = more robust, but slower.")
    # p.add_argument("-o", "--output", default="panorama.jpg", help="Output image (default: panorama.jpg)")
    p.add_argument("-o", "--output", help="Output image (default: output/panorama_datetime.jpg)")
    p.add_argument("--crop-ratio", type=float, default=0.70,
                   help="Keep the top fraction of each frame, 1.0 = no crop (default: 0.70)")
    p.add_argument("--max-width", type=int, default=1600,
                   help="Downscale frames wider than this, 0 = off (default: 1600)")
    p.add_argument("--nfeatures", type=int, default=8000, help="SIFT features per frame (default: 8000)")
    args = p.parse_args()

    if args.step < 1:
        p.error("--step must be >= 1")
    if not 0 < args.crop_ratio <= 1:
        p.error("--crop-ratio must be in (0, 1]")
    if not Path(args.video).is_file():
        p.error(f"Video not found: {args.video}")
    return args


def main():
    args = parse_args()
    try:
        frames = extract_frames(args.video, args.step, args.crop_ratio, args.max_width)
        if len(frames) < 2:
            raise RuntimeError("Need at least 2 frames (video too short or step too large).")
        pano = create_panorama(frames, args.nfeatures)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output = args.output
    else:
        today = datetime.today()
        formatted_date = str(today.strftime('%Y%m%d_%H%M%S'))
        
        os.makedirs("output", exist_ok=True)
    
        output = f"output/panorama_{formatted_date}.jpg"

    if not cv2.imwrite(output, pano):
        print(f"Could not write {output}", file=sys.stderr)
        sys.exit(1)
    print(f"\nSaved: {output}  ({pano.shape[1]} x {pano.shape[0]})")


if __name__ == "__main__":
    main()
