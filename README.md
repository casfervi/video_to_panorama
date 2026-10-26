# Video to Panorama

Command-line tool that builds a panorama (mosaic) from a video where the camera pans or moves across a scene. It samples frames from the video, aligns them with SIFT features and homographies, and blends them with feathered weights.

## How it works

1. **Frame extraction**: keeps one frame every `--step` frames, optionally crops the bottom of each frame and downscales wide frames.
2. **Feature detection**: SIFT keypoints and descriptors are computed once per frame.
3. **Matching**: FLANN matcher with Lowe's ratio test, then a robust homography (MAGSAC if available, RANSAC otherwise).
4. **Validation**: a homography is rejected if it has too few inliers, a low inlier ratio, an extreme scale or strong perspective terms. Frames that fail to align are dropped, and the next frame is matched against the last good one.
5. **Chaining**: homographies are chained and re-anchored on the middle frame to reduce distortion.
6. **Blending**: each frame is warped into its own bounding box on the canvas and blended with weights that fade toward the frame borders (feathering).
7. **Cropping**: the result is cropped to the area actually covered by frames.

## Requirements

- Python 3.8+
- `opencv-python` 4.5 or newer (SIFT is included in the main package since 4.4; MAGSAC needs 4.5+, otherwise it falls back to RANSAC)
- `numpy`

```bash
pip install opencv-python numpy
```

## Usage

```bash
python video_to_panorama.py VIDEO [-s STEP] [-o OUTPUT] [--crop-ratio R] [--max-width W] [--nfeatures N]
```

On Windows, quote paths that contain spaces:

```bat
python video_to_panorama.py "C:\Users\me\Downloads\video.mp4" --step 30
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `VIDEO` | required | Path to the input video. |
| `-s`, `--step` | `30` | Keep one frame every N frames. Smaller means more overlap and a more robust alignment, but slower. |
| `-o`, `--output` | `output/panorama_<YYYYMMDD_HHMMSS>.jpg` | Output image path. The `output/` folder is created if it does not exist. |
| `--crop-ratio` | `0.70` | Fraction of each frame to keep, measured from the top. `1.0` disables cropping. |
| `--max-width` | `1600` | Downscale frames wider than this many pixels. `0` disables downscaling. |
| `--nfeatures` | `8000` | Maximum number of SIFT features per frame. |

### Examples

```bash
# Defaults (one frame every 30, keep top 70%, save to output/)
python video_to_panorama.py video.mp4

# More overlap between frames, no crop, custom output name
python video_to_panorama.py video.mp4 -s 10 --crop-ratio 1.0 -o pano.jpg

# Full resolution, more features (slower, more memory)
python video_to_panorama.py video.mp4 --max-width 0 --nfeatures 12000
```

## Output

A JPEG image. The console prints per-frame keypoint counts, match and inlier statistics for each pair, which frames were used, and the final panorama size.

## Tuning and troubleshooting

- **"Could not align at least 2 frames"**: consecutive frames do not overlap enough. Lower `--step`.
- **Many frames rejected**: lower `--step`, or check that the scene has enough texture. Low-texture surfaces produce few reliable matches.
- **Ghosting or double edges**: usually caused by parallax or a non-planar scene (see limitations). A smaller `--step` can help a little.
- **Out of memory or very slow**: use a larger `--step`, a smaller `--max-width`, or fewer `--nfeatures`.
- **Unwanted objects at the bottom of the frame** (boat, hands, tripod): lower `--crop-ratio`.

## Limitations

- Works best when the scene is roughly planar, or when the camera rotates around its optical center. Strong parallax produces misalignment.
- Homographies are chained, so small errors accumulate over long sweeps. Re-anchoring on the middle frame reduces this but does not remove it.
- The canvas is capped at 80 megapixels. Larger panoramas are downscaled automatically. At that size the blending buffers use roughly 1.3 GB of RAM.
- Frames are held in memory during processing, so very long videos with a small `--step` need a lot of RAM.
- Moving objects in the scene may appear blurred or doubled in the blend.

## Tunable constants

These sit at the top of `video_to_panorama.py`:

| Constant | Meaning |
|---|---|
| `MAX_CANVAS_PIXELS` | Maximum panorama size in pixels before automatic downscaling. |
| `MIN_INLIERS` | Minimum RANSAC inliers to accept a frame pair. |
| `MIN_INLIER_RATIO` | Minimum inliers divided by good matches. |
| `MAX_PERSPECTIVE` | Largest allowed perspective term in a homography. |
| `SCALE_RANGE` | Allowed area scale change between consecutive frames. |
