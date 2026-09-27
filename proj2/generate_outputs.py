"""
Headless output generator for the Project 2 results page.

Mirrors the real algorithms in main.ipynb (convolution, finite differences,
DoG, unsharp masking, image alignment, hybrid images) but replaces the
notebook's interactive pieces (plt.ginput point-picking, cv2.imshow display
windows) with hardcoded correspondence points and file output, so it can run
headless and produce every image referenced by index.html.

Covers spec parts 1.1-2.2 only (2.3 / multiresolution blending is handled
separately in the notebook).
"""
import math
import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
import skimage.transform as sktr

DATA = 'data'
OUT = 'out'
os.makedirs(OUT, exist_ok=True)


def load_gray(name):
    return cv2.imread(os.path.join(DATA, name), cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0


def load_color(name):
    bgr = cv2.imread(os.path.join(DATA, name), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def save01(path, im):
    """Save a float image in [0, 1] (gray or RGB) as a file."""
    im = np.clip(im, 0, 1)
    if im.ndim == 2:
        cv2.imwrite(os.path.join(OUT, path), (im * 255).astype(np.uint8))
    else:
        bgr = cv2.cvtColor((im * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(OUT, path), bgr)


def normalize_for_display(im):
    im = np.abs(im)
    m = im.max()
    if m > 0:
        im = im / m * 255.0
    return im.astype(np.uint8)


# --------------------------------------------------------------------------
# Part 1.1: convolutions from scratch
# --------------------------------------------------------------------------

def convolve_4loops(im, kernel):
    img_h, img_w = im.shape
    ker_h, ker_w = kernel.shape
    kernel = np.flipud(np.fliplr(kernel))
    pad_h = (ker_h - 1) // 2
    pad_w = (ker_w - 1) // 2
    padded_im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros((img_h, img_w))
    for x in range(img_h):
        for y in range(img_w):
            for i in range(ker_h):
                for j in range(ker_w):
                    output[x, y] += padded_im[x + i, y + j] * kernel[i, j]
    return output


def convolve_2loops(im, kernel):
    img_h, img_w = im.shape
    ker_h, ker_w = kernel.shape
    kernel = np.flipud(np.fliplr(kernel))
    pad_h = (ker_h - 1) // 2
    pad_w = (ker_w - 1) // 2
    padded_im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros((img_h, img_w))
    for x in range(img_h):
        for y in range(img_w):
            region = padded_im[x:x + ker_h, y:y + ker_w]
            output[x, y] = np.sum(region * kernel)
    return output


def part1_1():
    print('=== Part 1.1 ===')
    im_selfie = load_gray('selfie.jpg')
    box_filter = np.ones((9, 9)) / 81.0

    t0 = time.time(); im_4loops = convolve_4loops(im_selfie, box_filter); t_4 = time.time() - t0
    t0 = time.time(); im_2loops = convolve_2loops(im_selfie, box_filter); t_2 = time.time() - t0
    t0 = time.time(); im_scipy = signal.convolve2d(im_selfie, box_filter, mode='same', boundary='fill', fillvalue=0); t_s = time.time() - t0

    match = np.allclose(im_4loops, im_2loops) and np.allclose(im_2loops, im_scipy)
    print(f'selfie shape: {im_selfie.shape}')
    print(f'4-loop:  {t_4:.3f}s')
    print(f'2-loop:  {t_2:.3f}s')
    print(f'scipy:   {t_s:.4f}s')
    print(f'all match: {match}')

    dx_kernel = np.array([-1, 0, 1], dtype=np.float64).reshape(1, 3)
    dy_kernel = np.array([-1, 0, 1], dtype=np.float64).reshape(3, 1)
    im_dx = convolve_2loops(im_selfie, dx_kernel)
    im_dy = convolve_2loops(im_selfie, dy_kernel)

    save01('1_1_selfie_original.jpg', im_selfie)
    save01('1_1_selfie_box.jpg', im_2loops)
    save01('1_1_selfie_dx.jpg', normalize_for_display(im_dx) / 255.0)
    save01('1_1_selfie_dy.jpg', normalize_for_display(im_dy) / 255.0)

    return dict(t_4=t_4, t_2=t_2, t_s=t_s, match=match, shape=im_selfie.shape)


# --------------------------------------------------------------------------
# Part 1.2: finite difference operator
# --------------------------------------------------------------------------

def part1_2():
    print('=== Part 1.2 ===')
    im_cameraman = load_gray('cameraman.png')
    dx_kernel = np.array([-1, 0, 1], dtype=np.float64).reshape(1, 3)
    dy_kernel = np.array([-1, 0, 1], dtype=np.float64).reshape(3, 1)
    im_dx = signal.convolve2d(im_cameraman, dx_kernel, mode='same', boundary='fill', fillvalue=0)
    im_dy = signal.convolve2d(im_cameraman, dy_kernel, mode='same', boundary='fill', fillvalue=0)
    im_grad_mag = np.sqrt(im_dx ** 2 + im_dy ** 2)

    save01('1_2_cameraman.jpg', im_cameraman)
    save01('1_2_dx.jpg', normalize_for_display(im_dx) / 255.0)
    save01('1_2_dy.jpg', normalize_for_display(im_dy) / 255.0)
    save01('1_2_grad_mag.jpg', normalize_for_display(im_grad_mag) / 255.0)

    for thresh, label in [(15, 'low'), (50, 'chosen'), (110, 'high')]:
        binarized = np.where(np.abs(im_grad_mag) > thresh / 255.0, 1.0, 0.0)
        save01(f'1_2_binarized_{label}.jpg', binarized)

    return im_cameraman, im_dx, im_dy, im_grad_mag


# --------------------------------------------------------------------------
# Part 1.3: DoG filters
# --------------------------------------------------------------------------

def kernel_preview(kernel, size=180):
    k = kernel.copy().astype(np.float64)
    k = k - k.min()
    if k.max() > 0:
        k = k / k.max()
    img = (k * 255).astype(np.uint8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def part1_3(im_cameraman, im_dx, im_dy, im_grad_mag):
    print('=== Part 1.3 ===')
    dx_kernel = np.array([-1, 0, 1], dtype=np.float64).reshape(1, 3)
    dy_kernel = np.array([-1, 0, 1], dtype=np.float64).reshape(3, 1)
    g1d = cv2.getGaussianKernel(ksize=9, sigma=1.5)
    gaussian_filter = g1d * g1d.T

    dog_x = signal.convolve2d(gaussian_filter, dx_kernel, mode='same', boundary='fill', fillvalue=0)
    dog_y = signal.convolve2d(gaussian_filter, dy_kernel, mode='same', boundary='fill', fillvalue=0)

    cv2.imwrite(os.path.join(OUT, '1_3_kernel_gaussian.png'), kernel_preview(gaussian_filter))
    cv2.imwrite(os.path.join(OUT, '1_3_kernel_dogx.png'), kernel_preview(dog_x))
    cv2.imwrite(os.path.join(OUT, '1_3_kernel_dogy.png'), kernel_preview(dog_y))

    im_blurred = signal.convolve2d(im_cameraman, gaussian_filter, mode='same', boundary='fill', fillvalue=0)
    im_blurred_dx = signal.convolve2d(im_blurred, dx_kernel, mode='same', boundary='fill', fillvalue=0)
    im_blurred_dy = signal.convolve2d(im_blurred, dy_kernel, mode='same', boundary='fill', fillvalue=0)
    im_blurred_grad_mag = np.sqrt(im_blurred_dx ** 2 + im_blurred_dy ** 2)
    binarized_blurred = np.where(np.abs(im_blurred_grad_mag) > 25 / 255.0, 1.0, 0.0)

    im_dogx_direct = signal.convolve2d(im_cameraman, dog_x, mode='same', boundary='fill', fillvalue=0)
    im_dogy_direct = signal.convolve2d(im_cameraman, dog_y, mode='same', boundary='fill', fillvalue=0)
    two_step_match = np.allclose(im_blurred_dx, im_dogx_direct, atol=1e-8) and \
        np.allclose(im_blurred_dy, im_dogy_direct, atol=1e-8)
    print('blur-then-diff == single DoG conv:', two_step_match)

    save01('1_3_blurred.jpg', im_blurred)
    save01('1_3_blurred_dx.jpg', normalize_for_display(im_blurred_dx) / 255.0)
    save01('1_3_blurred_dy.jpg', normalize_for_display(im_blurred_dy) / 255.0)
    save01('1_3_blurred_grad_mag.jpg', normalize_for_display(im_blurred_grad_mag) / 255.0)
    save01('1_3_blurred_binarized.jpg', binarized_blurred)
    save01('1_3_dogx_direct.jpg', normalize_for_display(im_dogx_direct) / 255.0)
    save01('1_3_dogy_direct.jpg', normalize_for_display(im_dogy_direct) / 255.0)

    return two_step_match


# --------------------------------------------------------------------------
# Bells & whistles: gradient orientation without arctan2 / cv2.phase
# --------------------------------------------------------------------------

def manual_gradient_orientation(dx, dy, n_bins=360):
    """Classify each pixel's (dx, dy) direction by nearest-angle match
    against a table of reference unit vectors built from sin/cos of known
    angles, instead of calling an inverse-trig 'angle' function like
    np.arctan2 or cv2.phase directly on (dx, dy)."""
    mag = np.sqrt(dx ** 2 + dy ** 2)
    safe_mag = np.where(mag > 1e-8, mag, 1.0)
    ux = (dx / safe_mag).ravel()
    uy = (dy / safe_mag).ravel()

    angles = np.arange(n_bins) * (2 * np.pi / n_bins)
    ref = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (n_bins, 2)

    pixel_vecs = np.stack([ux, uy], axis=1)  # (N, 2)
    dots = pixel_vecs @ ref.T  # (N, n_bins)
    best_bin = np.argmax(dots, axis=1)
    orientation_deg = (best_bin * (360.0 / n_bins)).reshape(dx.shape)
    orientation_deg[mag <= 1e-8] = 0
    return orientation_deg, mag


def part1_bonus(im_dx, im_dy):
    print('=== Bells & whistles: gradient orientation ===')
    orientation_deg, mag = manual_gradient_orientation(im_dx, im_dy)
    mag_norm = mag / (mag.max() + 1e-8)

    hsv = np.zeros((*orientation_deg.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (orientation_deg / 2).astype(np.uint8)  # OpenCV hue in [0,179]
    hsv[..., 1] = 255
    hsv[..., 2] = (np.clip(mag_norm * 3, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(OUT, '1_bonus_orientation.jpg'), bgr)

    # small hue legend wheel
    size = 200
    yy, xx = np.mgrid[0:size, 0:size] - size / 2
    r = np.sqrt(xx ** 2 + yy ** 2)
    wheel_orientation, _ = manual_gradient_orientation(xx, -yy)
    wheel_hsv = np.zeros((size, size, 3), dtype=np.uint8)
    wheel_hsv[..., 0] = (wheel_orientation / 2).astype(np.uint8)
    wheel_hsv[..., 1] = 255
    wheel_hsv[..., 2] = np.where(r <= size / 2, 255, 0).astype(np.uint8)
    wheel_bgr = cv2.cvtColor(wheel_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(os.path.join(OUT, '1_bonus_wheel.jpg'), wheel_bgr)


# --------------------------------------------------------------------------
# Part 2.1: unsharp masking
# --------------------------------------------------------------------------

def unsharp(im, a, sigma=2.0, ksize=9):
    g = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    impulse = signal.unit_impulse((ksize, ksize), 'mid')
    filt = (1 + a) * impulse - a * (g @ g.T)
    filt_3d = filt[:, :, np.newaxis]
    return np.clip(ndimage.convolve(im, filt_3d, mode='reflect'), 0, 1)


def gaussian_blur_color(im, sigma=2.0, ksize=9):
    g = cv2.getGaussianKernel(ksize=ksize, sigma=sigma)
    kernel_3d = (g @ g.T)[:, :, np.newaxis]
    return np.clip(ndimage.convolve(im, kernel_3d, mode='reflect'), 0, 1)


def part2_1():
    print('=== Part 2.1 ===')
    im_taj = load_color('taj.jpg')
    im_red = load_color('red.jpg')

    for name, im in [('taj', im_taj), ('red', im_red)]:
        blurred = gaussian_blur_color(im, sigma=2.0)
        high_freq = np.clip(im - blurred + 0.5, 0, 1)
        sharp = unsharp(im, a=5, sigma=2.0)
        save01(f'2_1_{name}_original.jpg', im)
        save01(f'2_1_{name}_blurred.jpg', blurred)
        save01(f'2_1_{name}_highfreq.jpg', high_freq)
        save01(f'2_1_{name}_sharp.jpg', sharp)

    for a in [1, 3, 6, 10]:
        sharp = unsharp(im_red, a=a, sigma=2.0)
        save01(f'2_1_red_a{a}.jpg', sharp)


# --------------------------------------------------------------------------
# Part 2.2: hybrid images (non-interactive alignment)
# --------------------------------------------------------------------------

def recenter(im, r, c):
    R, C = im.shape[:2]
    rpad = int(np.abs(2 * r + 1 - R))
    cpad = int(np.abs(2 * c + 1 - C))
    pad_width = [(0 if r > (R - 1) / 2 else rpad, 0 if r < (R - 1) / 2 else rpad),
                 (0 if c > (C - 1) / 2 else cpad, 0 if c < (C - 1) / 2 else cpad)]
    if im.ndim == 3:
        pad_width.append((0, 0))
    return np.pad(im, pad_width, mode='constant', constant_values=1.0)


def find_centers(p1, p2):
    cx = np.round(np.mean([p1[0], p2[0]]))
    cy = np.round(np.mean([p1[1], p2[1]]))
    return cx, cy


def align_image_centers(im1, im2, pts):
    p1, p2, p3, p4 = pts
    cx1, cy1 = find_centers(p1, p2)
    cx2, cy2 = find_centers(p3, p4)
    im1 = recenter(im1, cy1, cx1)
    im2 = recenter(im2, cy2, cx2)
    return im1, im2


def rescale_images(im1, im2, pts):
    p1, p2, p3, p4 = pts
    len1 = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    len2 = np.sqrt((p4[1] - p3[1]) ** 2 + (p4[0] - p3[0]) ** 2)
    dscale = len2 / len1
    channel_axis = -1 if im1.ndim == 3 else None
    if dscale < 1:
        im1 = sktr.rescale(im1, dscale, channel_axis=channel_axis)
    else:
        im2 = sktr.rescale(im2, 1. / dscale, channel_axis=channel_axis)
    return im1, im2


def rotate_im1(im1, pts):
    p1, p2, p3, p4 = pts
    theta1 = math.atan2(-(p2[1] - p1[1]), (p2[0] - p1[0]))
    theta2 = math.atan2(-(p4[1] - p3[1]), (p4[0] - p3[0]))
    dtheta = theta2 - theta1
    im1 = sktr.rotate(im1, dtheta * 180 / np.pi, mode='constant', cval=1.0)
    return im1, dtheta


def match_img_size(im1, im2):
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]
    if h1 < h2:
        im2 = im2[int(np.floor((h2 - h1) / 2.)): -int(np.ceil((h2 - h1) / 2.)), :]
    elif h1 > h2:
        im1 = im1[int(np.floor((h1 - h2) / 2.)): -int(np.ceil((h1 - h2) / 2.)), :]
    if w1 < w2:
        im2 = im2[:, int(np.floor((w2 - w1) / 2.)): -int(np.ceil((w2 - w1) / 2.))]
    elif w1 > w2:
        im1 = im1[:, int(np.floor((w1 - w2) / 2.)): -int(np.ceil((w1 - w2) / 2.))]
    assert im1.shape == im2.shape
    return im1, im2


def align_images(im1, im2, pts):
    im1, im2 = align_image_centers(im1, im2, pts)
    im1, im2 = rescale_images(im1, im2, pts)
    im1, _ = rotate_im1(im1, pts)
    im1, im2 = match_img_size(im1, im2)
    return im1, im2


def crop_blank_borders(im1, im2, thresh=0.92):
    """recenter()'s padding and rotate_im1()'s rotated corners both fill
    with white (constant_values=1.0 / cval=1.0), and identically on both
    images since they share a post-match_img_size shape -- match_img_size's
    crop can't remove it because it doesn't know which rows/cols are real
    content. Strip whole rows/cols off each edge while they're
    near-uniformly blank in *either* image."""
    h, w = im1.shape[:2]
    top, bottom, left, right = 0, h, 0, w

    def row_blank(im, r):
        return np.mean(im[r, :]) > thresh

    def col_blank(im, c):
        return np.mean(im[:, c]) > thresh

    while top < bottom - 1 and (row_blank(im1, top) or row_blank(im2, top)):
        top += 1
    while bottom > top + 1 and (row_blank(im1, bottom - 1) or row_blank(im2, bottom - 1)):
        bottom -= 1
    while left < right - 1 and (col_blank(im1, left) or col_blank(im2, left)):
        left += 1
    while right > left + 1 and (col_blank(im1, right - 1) or col_blank(im2, right - 1)):
        right -= 1

    return im1[top:bottom, left:right], im2[top:bottom, left:right]


def hybrid_image(im1, im2, sigma1, sigma2):
    """im1 is the low-frequency image, im2 is the high-frequency image.
    Kernel size scales with sigma (same convention as gaussian_stack in
    Part 2.3), unlike the notebook's fixed ksize=9 -- needed so larger
    sigmas actually blur instead of being truncated by a tiny kernel."""
    ksize1 = int(2 * np.ceil(3 * sigma1) + 1)
    ksize2 = int(2 * np.ceil(3 * sigma2) + 1)
    g1 = cv2.getGaussianKernel(ksize=ksize1, sigma=sigma1)
    g2 = cv2.getGaussianKernel(ksize=ksize2, sigma=sigma2)
    low_pass_im1 = cv2.filter2D(im1, -1, g1 @ g1.T)
    high_pass_im2 = im2 - cv2.filter2D(im2, -1, g2 @ g2.T)
    hybrid = np.clip(low_pass_im1 + high_pass_im2, 0, 1)
    return hybrid, low_pass_im1, high_pass_im2


def fft_spectrum(im):
    return np.log(np.abs(np.fft.fftshift(np.fft.fft2(im))) + 1e-8)


def save_spectrum(path, im):
    spectrum = fft_spectrum(im)
    plt.figure(figsize=(4, 4))
    plt.imshow(spectrum, cmap='gray')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(OUT, path), dpi=120, bbox_inches='tight', pad_inches=0)
    plt.close()


def downsample_max(im, max_dim=520):
    h, w = im.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return im


def run_hybrid_pair(name, a_name, a_pts, b_name, b_pts, low_source, sigma_low, sigma_high,
                     full_process=False):
    """Mirrors the notebook's per-pair calls exactly:
      a_aligned, b_aligned = align_images(a, b)   # a is rotate_im1's target
      hybrid = hybrid_image(<low_source>_aligned, <the other>_aligned, sigma_low, sigma_high)
    a/b are alignment roles (a gets rotated); low_source ('a' or 'b') is the
    separate, independent choice of which aligned image is the hybrid's
    low-frequency base -- the notebook's landscape/josh pair rotates
    landscape but uses josh as the low-frequency image, so these two
    choices don't always match.
    """
    print(f'=== Hybrid: {name} ===')
    a = load_gray(a_name)
    b = load_gray(b_name)

    pts = (*a_pts, *b_pts)
    a_aligned, b_aligned = align_images(a, b, pts)
    a_aligned, b_aligned = crop_blank_borders(a_aligned, b_aligned)
    a_aligned = downsample_max(a_aligned)
    b_aligned = downsample_max(b_aligned)

    low_aligned, high_aligned = (a_aligned, b_aligned) if low_source == 'a' else (b_aligned, a_aligned)
    low_name, high_name = (a_name, b_name) if low_source == 'a' else (b_name, a_name)

    save01(f'2_2_{name}_low_original.jpg', load_gray(low_name))
    save01(f'2_2_{name}_high_original.jpg', load_gray(high_name))
    save01(f'2_2_{name}_low_aligned.jpg', low_aligned)
    save01(f'2_2_{name}_high_aligned.jpg', high_aligned)

    hybrid, low_pass, high_pass = hybrid_image(low_aligned, high_aligned, sigma_low, sigma_high)
    save01(f'2_2_{name}_hybrid.jpg', hybrid)

    if full_process:
        save01(f'2_2_{name}_low_pass.jpg', np.clip(low_pass, 0, 1))
        save01(f'2_2_{name}_high_pass.jpg', np.clip(high_pass + 0.5, 0, 1))
        save_spectrum(f'2_2_{name}_fft_low_input.jpg', low_aligned)
        save_spectrum(f'2_2_{name}_fft_high_input.jpg', high_aligned)
        save_spectrum(f'2_2_{name}_fft_low_pass.jpg', low_pass)
        save_spectrum(f'2_2_{name}_fft_high_pass.jpg', high_pass)
        save_spectrum(f'2_2_{name}_fft_hybrid.jpg', hybrid)

    return hybrid


def part2_2():
    # Derek + Nutmeg -- required pair, shown with the full process.
    # notebook: align_images(nutmeg, derek) -- nutmeg (background margin to
    # spare) absorbs the rotation instead of Derek's edge-to-edge headshot.
    # hybrid_image(derek_aligned, nutmeg_aligned, sigma1=10.0, sigma2=5.0):
    # Derek is the low-frequency base, Nutmeg the high-frequency detail.
    derek_pts = ((293, 339), (441, 331))     # left eye, right eye
    nutmeg_pts = ((608, 283), (755, 357))    # cat's left eye, cat's right eye
    run_hybrid_pair('derek_nutmeg', 'nutmeg.jpg', nutmeg_pts, 'DerekPicture.jpg', derek_pts,
                     low_source='b', sigma_low=10.0, sigma_high=5.0, full_process=True)

    # Speed + Speed Smile. notebook: align_images(speed, speed_smile);
    # hybrid_image(speed_aligned, speed_smile_aligned, sigma1=1.5, sigma2=3)
    # -- Speed is the low-frequency base, Speed Smile the high-freq detail.
    speed_pts = ((661, 470), (930, 470))
    speed_smile_pts = ((177, 200), (264, 196))
    run_hybrid_pair('speed', 'speed.jpg', speed_pts, 'speed_smile.jpg', speed_smile_pts,
                     low_source='a', sigma_low=1.5, sigma_high=3.0)

    # Landscape + Josh. notebook: align_images(landscape, josh) -- landscape
    # absorbs the rotation -- but hybrid_image(josh_aligned,
    # landscape_aligned, sigma1=3, sigma2=5) makes JOSH the low-frequency
    # base and LANDSCAPE the high-frequency detail (inverted from the
    # alignment roles, and from the stale comment in the notebook).
    landscape_pts = ((204, 316), (521, 296))
    josh_pts = ((172, 133), (310, 116))
    run_hybrid_pair('landscape_josh', 'landscape.jpg', landscape_pts, 'josh.jpg', josh_pts,
                     low_source='b', sigma_low=3.0, sigma_high=5.0)


if __name__ == '__main__':
    timing = part1_1()
    im_cameraman, im_dx, im_dy, im_grad_mag = part1_2()
    two_step_match = part1_3(im_cameraman, im_dx, im_dy, im_grad_mag)
    part1_bonus(im_dx, im_dy)
    part2_1()
    part2_2()
    print('\nDone. timing =', timing, 'two_step_match =', two_step_match)
