# Panorama Tools

Small command-line tools for building and preparing panoramas with OpenCV.

| Script | Purpose |
|---|---|
| [`video_to_panorama.py`](#video_to_panoramapy) | Builds a panorama from a video where the camera pans or moves across a scene. |
| [`photos_to_panorama.py`](#photos_to_panoramapy) | Stitches a folder of overlapping photos into a panorama using OpenCV's `Stitcher`. |
| [`crop_image.py`](#crop_imagepy) | Crops an image to a rectangle given in pixel coordinates (for example, to trim a panorama). |

## Requirements

- Python 3.8+
- `opencv-python` 4.5 or newer (SIFT is in the main package since 4.4; MAGSAC needs 4.5+, otherwise `video_to_panorama.py` falls back to RANSAC)
- `numpy`
- `matplotlib` (only needed by `photos_to_panorama.py`)

```bash
pip install opencv-python numpy matplotlib
```

---

## video_to_panorama.py

Samples frames from a video, aligns them with SIFT features and homographies, and blends them with feathered weights.

### How it works

1. **Frame extraction**: keeps one frame every `--step` frames, optionally crops the bottom of each frame and downscales wide frames.
2. **Feature detection**: SIFT keypoints and descriptors are computed once per frame.
3. **Matching**: FLANN matcher with Lowe's ratio test, then a robust homography (MAGSAC if available, RANSAC otherwise).
4. **Validation**: a homography is rejected if it has too few inliers, a low inlier ratio, an extreme scale or strong perspective terms. Frames that fail to align are dropped, and the next frame is matched against the last good one.
5. **Chaining**: homographies are chained and re-anchored on the middle frame to reduce distortion.
6. **Blending**: each frame is warped into its own bounding box on the canvas and blended with weights that fade toward the frame borders (feathering).
7. **Cropping**: the result is cropped to the area actually covered by frames.

### Usage

```bash
python video_to_panorama.py VIDEO [-s STEP] [-o OUTPUT] [--crop-ratio R] [--max-width W] [--nfeatures N]
```

On Windows, quote paths that contain spaces:

```bat
python video_to_panorama.py "C:\Users\me\Downloads\video.mp4" --step 30
```

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

### Output

A JPEG image. The console prints per-frame keypoint counts, match and inlier statistics for each pair, which frames were used, and the final panorama size.

### Tuning and troubleshooting

- **"Could not align at least 2 frames"**: consecutive frames do not overlap enough. Lower `--step`.
- **Many frames rejected**: lower `--step`, or check that the scene has enough texture. Low-texture surfaces produce few reliable matches.
- **Ghosting or double edges**: usually caused by parallax or a non-planar scene (see limitations). A smaller `--step` can help a little.
- **Out of memory or very slow**: use a larger `--step`, a smaller `--max-width`, or fewer `--nfeatures`.
- **Unwanted objects at the bottom of the frame** (boat, hands, tripod): lower `--crop-ratio`.

### Limitations

- Works best when the scene is roughly planar, or when the camera rotates around its optical center. Strong parallax produces misalignment.
- Homographies are chained, so small errors accumulate over long sweeps. Re-anchoring on the middle frame reduces this but does not remove it.
- The canvas is capped at 80 megapixels. Larger panoramas are downscaled automatically. At that size the blending buffers use roughly 1.3 GB of RAM.
- Frames are held in memory during processing, so very long videos with a small `--step` need a lot of RAM.
- Moving objects in the scene may appear blurred or doubled in the blend.

### Tunable constants

These sit at the top of `video_to_panorama.py`:

| Constant | Meaning |
|---|---|
| `MAX_CANVAS_PIXELS` | Maximum panorama size in pixels before automatic downscaling. |
| `MIN_INLIERS` | Minimum RANSAC inliers to accept a frame pair. |
| `MIN_INLIER_RATIO` | Minimum inliers divided by good matches. |
| `MAX_PERSPECTIVE` | Largest allowed perspective term in a homography. |
| `SCALE_RANGE` | Allowed area scale change between consecutive frames. |

---

## photos_to_panorama.py

Stitches all the photos in a folder into one panorama with OpenCV's high-level `Stitcher`, using `SCANS` mode. This mode uses an affine model and is meant for flat scenes captured by a camera that moves along them (walls, surfaces, documents), rather than a camera that rotates in place. Use it when you have overlapping still photos instead of a video.

### How it works

1. Reads every file in the folder, sorted by file name.
2. The `Stitcher` finds features, matches the photos, estimates how they fit together, compensates exposure and blends the overlaps.
3. Saves the result as a JPEG.

### Usage

```bash
python photos_to_panorama.py --image_folder FOLDER [--output_path OUTPUT_FOLDER] [--show_img True]
```

| Argument | Default | Description |
|---|---|---|
| `--image_folder` | required | Folder with the photos to stitch. |
| `--output_path` | `output` | Output folder. It is created if it does not exist. |
| `--show_img` | off | Pass any value (for example `True`) to show a grid of the input photos and the result in matplotlib windows. Leave it out to disable. |

### Examples

```bash
# Stitch every photo in a folder, save to output/
python photos_to_panorama.py --image_folder path/to/photos

# Custom output folder, with preview windows
python photos_to_panorama.py --image_folder path/to/photos --output_path results --show_img True
```

### Output

The panorama is saved as `<image folder name>_stitched.jpg` inside the output folder. For example, photos in `trip_photos/` produce `output/trip_photos_stitched.jpg`. The console lists each file read and how many images were found.

### Requirements for the photos

- The folder should contain only image files. Every file in it is read, and a file that is not an image causes an error.
- At least 2 photos are needed, and an empty folder makes the script fail.
- Neighboring photos need a good overlap (roughly 30% or more) and enough texture to match.
- Keep exposure, focus and zoom as similar as possible between shots.
- Photos are processed in file name order, so numbered names (`img_001.jpg`, `img_002.jpg`, ...) keep things predictable.

### Troubleshooting

- **The script prints only `Error!`**: stitching failed. The usual cause is too little overlap or too few matching features between some photos. Retake with more overlap, or remove photos that do not overlap with the rest.
- **Result looks distorted or bent**: the scene is probably not flat, or the camera rotated instead of moving along the scene. In the script, switch to the commented-out line `cv2.Stitcher_create(cv2.STITCHER_PANORAMA)`, which uses a rotating-camera model.
- **Black borders around the result**: the output is not cropped automatically. Trim it with [`crop_image.py`](#crop_imagepy).
- **Slow or out of memory**: resize the photos before stitching. Many full-resolution photos need a lot of RAM.

---

## crop_image.py

Crops a single image to a rectangle defined by pixel coordinates and saves the result as a JPEG. It also prints the original and cropped sizes and how much of the image was removed.

### Usage

```bash
python crop_image.py --image IMAGE --crop_x X --crop_w X_END --crop_y Y --crop_h Y_END [--output_path FOLDER]
```

| Argument | Default | Description |
|---|---|---|
| `--image` | required | Path to the image to crop. |
| `--crop_x` | required | Start of the crop in X (left edge, in pixels). |
| `--crop_w` | required | **End** of the crop in X (right edge, in pixels). This is a coordinate, not a width. |
| `--crop_y` | required | Start of the crop in Y (top edge, in pixels). |
| `--crop_h` | required | **End** of the crop in Y (bottom edge, in pixels). This is a coordinate, not a height. |
| `--output_path` | `output` | Output folder. It is created if it does not exist. |

Coordinates are in pixels with the origin at the top-left corner. Internally the crop is `img[crop_y:crop_h, crop_x:crop_w]`, so the resulting size is `(crop_w - crop_x)` by `(crop_h - crop_y)`.

### Examples

Vertical image:

```bash
python crop_image.py --image path_to_image\image.jpg --crop_x 200 --crop_w 960 --crop_y 150 --crop_h 3400 --output_path folder_here
```

Horizontal image:

```bash
python crop_image.py --image path_to_image\image.jpg --crop_x 150 --crop_w 3400 --crop_y 200 --crop_h 960 --output_path folder_here
```

### Output

The cropped image is saved as `<original name>_croped.jpg` inside the output folder. For example, `pano.jpg` becomes `output/pano_croped.jpg`.

### Notes

- Always check that each end value is larger than its start value. Otherwise the crop is empty and nothing is saved.
- Values larger than the image size are clipped to the image border.
- The output is always saved as JPEG, regardless of the input format.

---

## Example workflows

From a video:

```bash
# 1. Build the panorama from a video
python video_to_panorama.py video.mp4 -s 15 -o output/pano.jpg

# 2. Trim unwanted borders from the result
python crop_image.py --image output/pano.jpg --crop_x 100 --crop_w 3000 --crop_y 50 --crop_h 900
```

From photos:

```bash
# 1. Stitch a folder of overlapping photos (saved as output/<folder name>_stitched.jpg)
python photos_to_panorama.py --image_folder path/to/photos

# 2. Trim unwanted borders from the result
python crop_image.py --image output/photos_stitched.jpg --crop_x 100 --crop_w 3000 --crop_y 50 --crop_h 900
```
