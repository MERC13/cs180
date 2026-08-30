"""Combine the dolly-zoom stills (1.webp..8.webp) into a looping animated GIF.

Re-run this script any time the source stills change:
    python make_gif.py
"""

from pathlib import Path
from PIL import Image

HERE = Path(__file__).parent
FRAME_PATHS = [HERE / f"{i}.webp" for i in range(1, 9)]
OUTPUT_PATH = HERE / "dolly_zoom.gif"

MAX_WIDTH = 560
FRAME_DURATION_MS = 150


def load_frames():
    frames = [Image.open(p).convert("RGB") for p in FRAME_PATHS]

    # Normalize every frame to the same size so the GIF canvas doesn't jitter.
    target_w = min(MAX_WIDTH, min(f.width for f in frames))
    target_h = round(frames[0].height * (target_w / frames[0].width))
    frames = [f.resize((target_w, target_h), Image.LANCZOS) for f in frames]
    return frames


def main():
    frames = load_frames()

    # Ping-pong (1..8..1) instead of a hard cut from the last frame back to
    # the first, so the loop reads as a continuous "breathing" dolly zoom.
    sequence = frames + frames[-2:0:-1]

    sequence[0].save(
        OUTPUT_PATH,
        format="GIF",
        save_all=True,
        append_images=sequence[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.1f} KB, {len(sequence)} frames)")


if __name__ == "__main__":
    main()
