"""
Headless output generator for the Project 2 results page.

Mirrors the real algorithms in main.ipynb (convolution, finite differences,
DoG, unsharp masking, image alignment, hybrid images) but replaces the
notebook's interactive pieces (plt.ginput point-picking, cv2.imshow display
windows) with hardcoded correspondence points and file output, so it can run
headless and produce every image referenced by index.html.

Covers spec parts 1.1-2.4.
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


def normalize_signed_for_display(im):
    """Map a signed derivative to a grey display: 0 -> mid-grey (128),
    scaled symmetrically by the largest-magnitude value so sign survives --
    unlike normalize_for_display's np.abs(), which maps both a rising and a
    falling edge to the same bright value and flattens everything else to
    black."""
    m = np.abs(im).max()
    if m > 0:
        im = im / m * 127.0
    return np.clip(im + 128.0, 0, 255).astype(np.uint8)


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
    save01('1_1_selfie_dx.jpg', normalize_signed_for_display(im_dx) / 255.0)
    save01('1_1_selfie_dy.jpg', normalize_signed_for_display(im_dy) / 255.0)

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
    save01('1_2_dx.jpg', normalize_signed_for_display(im_dx) / 255.0)
    save01('1_2_dy.jpg', normalize_signed_for_display(im_dy) / 255.0)
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
    diff_dx = np.abs(im_blurred_dx - im_dogx_direct)
    border = 6
    interior_diff = diff_dx[border:-border, border:-border]
    print('blur-then-diff == single DoG conv:', two_step_match)
    print('  full-image max diff:', diff_dx.max())
    print(f'  interior (excl. {border}px border) max diff:', interior_diff.max())

    save01('1_3_blurred.jpg', im_blurred)
    save01('1_3_blurred_dx.jpg', normalize_signed_for_display(im_blurred_dx) / 255.0)
    save01('1_3_blurred_dy.jpg', normalize_signed_for_display(im_blurred_dy) / 255.0)
    save01('1_3_blurred_grad_mag.jpg', normalize_for_display(im_blurred_grad_mag) / 255.0)
    save01('1_3_blurred_binarized.jpg', binarized_blurred)
    save01('1_3_dogx_direct.jpg', normalize_signed_for_display(im_dogx_direct) / 255.0)
    save01('1_3_dogy_direct.jpg', normalize_signed_for_display(im_dogy_direct) / 255.0)

    return two_step_match


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

    im_red_blurred = gaussian_blur_color(im_red, sigma=2.0)
    im_red_highfreq = np.clip(im_red - im_red_blurred + 0.5, 0, 1)
    im_red_sharp = unsharp(im_red, a=5, sigma=2.0)
    save01('2_1_red_original.jpg', im_red)
    save01('2_1_red_blurred.jpg', im_red_blurred)
    save01('2_1_red_highfreq.jpg', im_red_highfreq)
    save01('2_1_red_sharp.jpg', im_red_sharp)

    im_taj_blurred = gaussian_blur_color(im_taj, sigma=2.0)
    im_taj_highfreq = np.clip(im_taj - im_taj_blurred + 0.5, 0, 1)
    im_taj_sharp = unsharp(im_taj, a=5, sigma=2.0)
    save01('2_1_taj_original.jpg', im_taj)
    save01('2_1_taj_blurred.jpg', im_taj_blurred)
    save01('2_1_taj_highfreq.jpg', im_taj_highfreq)
    save01('2_1_taj_sharp.jpg', im_taj_sharp)

    for a in [1, 3, 6, 10]:
        save01(f'2_1_red_a{a}.jpg', unsharp(im_red, a=a, sigma=2.0))

    # Evaluation: blur a sharp image, then run the same unsharp filter on
    # the blurred version and see how much of the original detail it
    # actually recovers.
    im_sunset = load_color('sunset.jpg')
    im_sunset_blurred = gaussian_blur_color(im_sunset, sigma=2.0)
    im_sunset_reconstruct = unsharp(im_sunset_blurred, a=5, sigma=2.0)
    save01('2_1_sunset_original.jpg', im_sunset)
    save01('2_1_sunset_blurred.jpg', im_sunset_blurred)
    save01('2_1_sunset_reconstruct.jpg', im_sunset_reconstruct)


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
    Part 2.3) so larger sigmas actually blur instead of being truncated
    by a too-small kernel."""
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
    # Derek + Nutmeg -- required pair, shown as originals + final hybrid only
    # (the "favorite result" with the full process is Landscape + Josh below).
    # notebook: align_images(nutmeg, derek) -- nutmeg (background margin to
    # spare) absorbs the rotation instead of Derek's edge-to-edge headshot.
    # hybrid_image(derek_aligned, nutmeg_aligned, sigma1=10.0, sigma2=5.0):
    # Derek is the low-frequency base, Nutmeg the high-frequency detail.
    derek_pts = ((293, 339), (441, 331))     # left eye, right eye
    nutmeg_pts = ((608, 283), (755, 357))    # cat's left eye, cat's right eye
    run_hybrid_pair('derek_nutmeg', 'nutmeg.jpg', nutmeg_pts, 'DerekPicture.jpg', derek_pts,
                     low_source='b', sigma_low=10.0, sigma_high=5.0)

    # Speed + Speed Smile. notebook: align_images(speed, speed_smile);
    # hybrid_image(speed_aligned, speed_smile_aligned, sigma1=1.5, sigma2=3)
    # -- Speed is the low-frequency base, Speed Smile the high-freq detail.
    speed_pts = ((661, 470), (930, 470))
    speed_smile_pts = ((177, 200), (264, 196))
    run_hybrid_pair('speed', 'speed.jpg', speed_pts, 'speed_smile.jpg', speed_smile_pts,
                     low_source='a', sigma_low=1.5, sigma_high=3.0)

    # Landscape + Josh -- favorite result, shown with the full process.
    # notebook: align_images(landscape, josh) -- landscape absorbs the
    # rotation -- but hybrid_image(josh_aligned, landscape_aligned,
    # sigma1=3, sigma2=5) makes JOSH the low-frequency base and LANDSCAPE
    # the high-frequency detail (inverted from the alignment roles).
    landscape_pts = ((204, 316), (521, 296))
    josh_pts = ((172, 133), (310, 116))
    run_hybrid_pair('landscape_josh', 'landscape.jpg', landscape_pts, 'josh.jpg', josh_pts,
                     low_source='b', sigma_low=3.0, sigma_high=5.0, full_process=True)


# --------------------------------------------------------------------------
# Part 2.3: Gaussian and Laplacian stacks
# --------------------------------------------------------------------------

def gaussian_stack(im, levels, sigma=2.0):
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    g = cv2.getGaussianKernel(ksize, sigma)
    kernel = g @ g.T
    stack = [im.astype(np.float64)]
    for _ in range(1, levels):
        stack.append(cv2.filter2D(stack[-1], -1, kernel))
    return stack


def laplacian_stack(g_stack):
    l_stack = [g_stack[i] - g_stack[i + 1] for i in range(len(g_stack) - 1)]
    l_stack.append(g_stack[-1])
    return l_stack


def multires_blend(im1, im2, mask, levels=6, sigma=8.0):
    g1 = gaussian_stack(im1, levels, sigma)
    g2 = gaussian_stack(im2, levels, sigma)
    gm = gaussian_stack(mask.astype(np.float64), levels, sigma)
    l1 = laplacian_stack(g1)
    l2 = laplacian_stack(g2)
    blended_stack = []
    for i in range(levels):
        m = gm[i]
        if im1.ndim == 3 and m.ndim == 2:
            m = m[:, :, np.newaxis]
        blended_stack.append(m * l1[i] + (1 - m) * l2[i])
    blended = np.clip(np.sum(blended_stack, axis=0), 0, 1)
    return blended, l1, l2, blended_stack, gm


def save_laplacian_grid(path, l1, l2, gm, blended_lstack, col_titles):
    """col_titles labels three columns per level: m*l1 (im1's masked
    contribution), (1-m)*l2 (im2's masked contribution), and their sum
    (the actual blended level) -- not the raw, unmasked Laplacian level of
    each source image."""
    levels = len(l1)
    fig, axes = plt.subplots(levels, 3, figsize=(9, 3 * levels))
    for i in range(levels):
        m = gm[i]
        if l1[i].ndim == 3 and m.ndim == 2:
            m = m[:, :, np.newaxis]
        cols = (m * l1[i], (1 - m) * l2[i], blended_lstack[i])
        for ax, im, col_title in zip(axes[i], cols, col_titles):
            # 0 -> mid-grey, not black -- matches how Fig 3.42 shows these.
            disp = im if i == levels - 1 else normalize_signed_for_display(im).astype(np.float64) / 255.
            ax.imshow(np.clip(disp, 0, 1))
            ax.axis('off')
            if i == 0:
                ax.set_title(col_title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, path), dpi=110)
    plt.close(fig)


def part2_3():
    print('=== Part 2.3 ===')
    apple = load_color('apple.jpg')
    orange = load_color('orange.jpg')
    assert apple.shape == orange.shape, 'apple.jpg and orange.jpg must be the same size'

    h, w = apple.shape[:2]
    mask = np.zeros((h, w), dtype=np.float64)
    mask[:, :w // 2] = 1.0

    n_levels = 6
    oraple, apple_lstack, orange_lstack, blended_lstack, mask_gstack = multires_blend(
        apple, orange, mask, n_levels, sigma=8.0)

    save01('2_3_apple.jpg', apple)
    save01('2_3_orange.jpg', orange)
    save01('2_3_oraple.jpg', oraple)
    save_laplacian_grid('2_3_fig342.jpg', apple_lstack, orange_lstack, mask_gstack, blended_lstack,
                         ('Apple × mask', 'Orange × (1 − mask)', 'Blended'))


# --------------------------------------------------------------------------
# Part 2.4: Multiresolution blending
# --------------------------------------------------------------------------

def center_crop_to_ratio(im, target_ratio):
    h, w = im.shape[:2]
    if w / h > target_ratio:
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        im = im[:, x0:x0 + new_w]
    else:
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        im = im[y0:y0 + new_h, :]
    return im


def match_size(im1, im2):
    h1, w1 = im1.shape[:2]
    im2 = center_crop_to_ratio(im2, w1 / h1)
    im2 = cv2.resize(im2, (w1, h1), interpolation=cv2.INTER_AREA)
    return im1, im2


def align_circle_to(src, src_center, src_radius, dst_shape, dst_center, dst_radius, border=(0, 0, 0)):
    """Rescale and recenter src (scale + translation, keyed on a circle's
    center/radius instead of an eye pair) so its disc lands exactly on
    dst_center at dst_radius, on a canvas the size of dst_shape. Without
    this, the sun and moon photos -- shot at different focal
    lengths/crops -- just don't share a common apparent size, so no mask,
    however well blended, hides one disc being visibly bigger than the
    other."""
    scale = dst_radius / src_radius
    M = np.array([[scale, 0, dst_center[0] - scale * src_center[0]],
                  [0, scale, dst_center[1] - scale * src_center[1]]], dtype=np.float64)
    h, w = dst_shape[:2]
    aligned = cv2.warpAffine((src * 255).astype(np.uint8), M, (w, h), borderValue=border)
    return aligned.astype(np.float64) / 255.


def part2_4():
    print('=== Part 2.4 ===')
    n_levels = 6

    # Irregular mask #1 -- favorite result. Sun + Moon, split by a
    # DIAGONAL seam through the shared disc center -- not a line straight
    # across the whole frame (the line only cuts the disc; both flanking
    # backgrounds are already matching black, so there's nothing to blend
    # out there).
    #
    # Both photos put their disc at a different pixel radius (moon.jpg's
    # disc is ~148px on a 300x300 canvas; star.jpg's sun is ~132px on a
    # 340x340 canvas -- measured with cv2.HoughCircles), so the moon is
    # rescaled and recentered onto the sun's canvas via align_circle_to
    # first. Skipping this leaves the two discs different apparent sizes
    # no matter how the mask is blended.
    sun = load_color('star.jpg')
    moon = load_color('moon.jpg')
    moon_aligned = align_circle_to(moon, (150, 150), 148, sun.shape, (168, 169), 132)

    h, w = sun.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = 168, 169
    sunmoon_mask = ((xx - cx) - (yy - cy) > 0).astype(np.float64)  # 1 = sun

    sunmoon_blend, sunmoon_l1, sunmoon_l2, sunmoon_lstack, sunmoon_mask_gstack = multires_blend(
        sun, moon_aligned, sunmoon_mask, 8, sigma=30.0)

    save01('2_4_sunmoon_sun.jpg', sun)
    save01('2_4_sunmoon_moon.jpg', moon_aligned)
    save01('2_4_sunmoon_mask.jpg', np.repeat(sunmoon_mask[:, :, np.newaxis], 3, axis=2))
    save01('2_4_sunmoon_blend.jpg', sunmoon_blend)

    sunmoon_mask_3d = sunmoon_mask[:, :, np.newaxis]
    save01('2_4_sunmoon_sun_masked.jpg', sun * sunmoon_mask_3d)
    save01('2_4_sunmoon_moon_masked.jpg', moon_aligned * (1 - sunmoon_mask_3d))
    save_laplacian_grid('2_4_sunmoon_lstack.jpg', sunmoon_l1, sunmoon_l2, sunmoon_mask_gstack, sunmoon_lstack,
                         ('Sun × mask', 'Moon × (1 − mask)', 'Blended'))

    # Irregular mask #2 -- a unique mask for seasonal_trees.jpg: instead
    # of a synthetic geometric shape (ellipse, disc, band), the mask is
    # the actual canopy SILHOUETTE of the summer tree, extracted by
    # thresholding its own green hue in HSV. Wherever that organic,
    # content-derived shape falls, the summer canopy shows through;
    # everywhere else -- sky, ground, gaps between branches -- shows the
    # same tree's snow-covered winter self.
    #
    # seasonal_trees.jpg is itself a 2x2 collage of one tree across four
    # seasons; the winter (bottom-left) and summer (bottom-right) panels
    # are cropped out here as the two source images.
    collage = cv2.imread(os.path.join(DATA, 'seasonal_trees.jpg'))
    ch, cw = collage.shape[:2]
    hh, hw = ch // 2, cw // 2
    winter_bgr = collage[hh:ch, 0:hw]
    summer_bgr = collage[hh:ch, hw:cw]
    th = min(winter_bgr.shape[0], summer_bgr.shape[0])
    tw = min(winter_bgr.shape[1], summer_bgr.shape[1])
    winter_bgr, summer_bgr = winter_bgr[:th, :tw], summer_bgr[:th, :tw]
    winter = cv2.cvtColor(winter_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.
    summer = cv2.cvtColor(summer_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.

    summer_hsv = cv2.cvtColor(summer_bgr, cv2.COLOR_BGR2HSV)
    hue, sat, _ = cv2.split(summer_hsv)
    canopy_mask = ((hue > 30) & (hue < 95) & (sat > 40)).astype(np.uint8) * 255
    morph_kernel = np.ones((5, 5), np.uint8)
    canopy_mask = cv2.morphologyEx(canopy_mask, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
    canopy_mask = cv2.morphologyEx(canopy_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    canopy_mask = (canopy_mask > 127).astype(np.float64)
    canopy_mask[160:, :] = 0.0  # exclude the green grass field -- canopy only

    tree_blend, _, _, _, _ = multires_blend(summer, winter, canopy_mask, n_levels, sigma=6.0)

    save01('2_4_tree_winter.jpg', winter)
    save01('2_4_tree_summer.jpg', summer)
    save01('2_4_tree_mask.jpg', np.repeat(canopy_mask[:, :, np.newaxis], 3, axis=2))
    save01('2_4_tree_blend.jpg', tree_blend)


if __name__ == '__main__':
    timing = part1_1()
    im_cameraman, im_dx, im_dy, im_grad_mag = part1_2()
    two_step_match = part1_3(im_cameraman, im_dx, im_dy, im_grad_mag)
    part2_1()
    part2_2()
    part2_3()
    part2_4()
    print('\nDone. timing =', timing, 'two_step_match =', two_step_match)
