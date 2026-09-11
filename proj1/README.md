# CS180 Project 1: Images of the Russian Empire

Colorizes Prokudin-Gorskii glass-plate scans by splitting each scan into
its blue/green/red exposures and aligning green and red onto blue (using
brute-force search for small .jpg scans, and a coarse-to-fine image
pyramid for the large .tif scans).

## Setup

```
pip install -r requirements.txt
```

Requires the `data/` folder (provided source scans, `.jpg`/`.tif`) to sit
alongside `main.py`.

## Running

```
python main.py
```

This runs three stages in order:

1. **Single-scale** alignment on the small `.jpg` scans (`cathedral`,
   `monastery`, `tobolsk`).
2. **Multi-scale (pyramid)** alignment on the rest of the provided `.tif`/
   `.jpg` scans.
3. **Multi-scale (pyramid)** alignment on a few extra, student-chosen
   images (`beans.jpg`, `ceramic.jpg`, `woman.jpg`).

For each image, a window pops up showing the aligned result — press any
key to close it and move on to the next image. The final aligned color
image for each scan is written to `out/`, and the computed (x, y) shifts
for the green and red channels are printed to the console.
