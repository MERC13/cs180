# CS 180 Project 2 — Fun with Filters and Frequencies

`main.ipynb` is a single notebook that implements every part of the
assignment in order, top to bottom:

| Part | What it does |
|---|---|
| 1.1 | Convolution from scratch (4-loop and 2-loop versions), checked against `scipy.signal.convolve2d` on a box filter and finite-difference kernels |
| 1.2 | Finite-difference edge detection (partial derivatives, gradient magnitude, binarized edges) on the cameraman image |
| 1.3 | Derivative-of-Gaussian filters — blur-then-differentiate vs. a single DoG convolution, verified to agree |
| 2.1 | Unsharp masking (image sharpening), including a blur-then-resharpen evaluation |
| 2.2 | Hybrid images (Oliva/Torralba/Schyns) — low-pass one photo, high-pass another, combine |
| 2.3 | Gaussian and Laplacian stacks (pyramids without downsampling), recreating Szeliski Fig. 3.42 with an apple/orange blend |
| 2.4 | Multiresolution blending (Burt & Adelson) with irregular masks — a sun/moon eclipse and a tree that's summer in one half, winter in the other |

Each part is its own markdown-separated section with the code immediately
followed by `show_fit(...)` calls that display the results in a window.

## Requirements

```
numpy
opencv-python
matplotlib
scipy
scikit-image
```

The first cell (`%pip install ...`) installs these automatically if
they're missing.

## Data files

The notebook expects a `data/` folder next to it, containing:

```
apple.jpg, cameraman.png, DerekPicture.jpg, josh.jpg, landscape.jpg,
moon.jpg, nutmeg.jpg, orange.jpg, red.jpg, seasonal_trees.jpg,
selfie.jpg, speed.jpg, speed_smile.jpg, star.jpg, sunset.jpg, taj.jpg
```

`selfie.jpg` is a personal photo — substitute your own grayscale
self-portrait if you don't have it. Every other file is a plain
photograph (no special preprocessing needed); the notebook handles
resizing, cropping, and format conversion itself.

## How to reproduce the results

1. Put the notebook and the `data/` folder in the same directory.
2. Open `main.ipynb` in Jupyter, JupyterLab, or VS Code's notebook UI.
3. Run cells **in order, top to bottom** — later parts reuse functions
   (`gaussian_stack`, `multires_blend`, `align_images`, etc.) and
   variables defined in earlier cells.
4. Most cells open a window per result via OpenCV
   (`cv2.imshow`/`show_fit`) — press any key to close each window and
   let the next one open. Part 2.3/2.4's stack visualizations use
   matplotlib figures instead and just render inline.
5. **Part 2.2 only** requires manual interaction: `align_images()` pops
   up each photo pair and asks you to click 2 corresponding points
   (we used both eyes) per image — click one point, then the second, on
   the first photo, then repeat on the second. This picks the alignment
   (center, scale, rotation) between the pair. The rest of the notebook
   runs unattended.

Everything downstream of the raw pixel arrays is pure NumPy/OpenCV/SciPy
— no saved intermediate files are required to reproduce a result from
scratch; re-running a cell recomputes it from the source images.
