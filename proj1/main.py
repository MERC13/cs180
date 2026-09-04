import numpy as np
import cv2
import glob
import os

# ====== UTILS ====== #

def crop(a, frac=0.1):
    h, w = a.shape
    dh, dw = int(h*frac), int(w*frac)
    return a[dh:h-dh, dw:w-dw]

def show_fit(window_name, img, max_dim=900):
    h, w = img.shape[:2]
    scale = min(max_dim / h, max_dim / w, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(window_name, img)

def grad_mag(im):
    gx = cv2.Sobel(im.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(im.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx ** 2 + gy ** 2)

def align(im1, im2, metric, window=15):
    best_shift = (0, 0)
    best_score = float('inf') if metric == "L2" else -float('inf')
    shift_range = range(-window, window + 1)
    h, w = im1.shape

    # score on gradient magnitude rather than raw intensity: far less
    # sensitive to per-channel brightness differences, and better at telling
    # apart visually-similar repeated structures (e.g. matching domes) than
    # raw pixel values are
    grad1 = grad_mag(im1)
    grad2 = grad_mag(im2)

    for dx in shift_range:
        for dy in shift_range:
            shifted_grad1 = np.roll(grad1, shift=(dy, dx), axis=(0, 1))

            # exclude exactly the rows/columns this shift wrapped in from the
            # opposite edge, so wraparound garbage never enters the score
            my, mx = abs(dy), abs(dx)
            s1 = shifted_grad1[my:h - my, mx:w - mx] if (my or mx) else shifted_grad1
            s2 = grad2[my:h - my, mx:w - mx] if (my or mx) else grad2

            if metric == "L2":
                # mean, not sum: candidates get compared over different-sized
                # regions now, so the score must be per-pixel to stay comparable
                score = np.sqrt(np.mean((s1.astype(np.float64) - s2) ** 2))
                if score < best_score:
                    best_score = score
                    best_shift = (dy, dx)
            elif metric == "NCC":
                mean_im1 = s1 - s1.mean()
                mean_im2 = s2 - s2.mean()
                score = np.sum(mean_im1 * mean_im2) / (np.linalg.norm(mean_im1) * np.linalg.norm(mean_im2))

                if score > best_score:
                    best_score = score
                    best_shift = (dy, dx)

    return np.roll(im1, shift=best_shift, axis=(0, 1)), best_shift

def pyramid_search(im1, im2, metric, scale=None, window=15):
    if scale is None:
        # pick a starting (coarsest) scale so that level isn't degenerately
        # tiny on small images, but never coarser than 1/16 (tuned for the
        # multi-thousand-pixel scans)
        scale = 1.0
        while min(im1.shape) * scale > 150 and scale > 0.0625:
            scale /= 2

    if scale == 1:
        return align(im1, im2, metric, window=window)
    else:
        im1_small = cv2.resize(im1, (int(im1.shape[1] * scale), int(im1.shape[0] * scale)))
        im2_small = cv2.resize(im2, (int(im2.shape[1] * scale), int(im2.shape[0] * scale)))

        _, shift_small = align(im1_small, im2_small, metric, window)
        shift_here = (np.array(shift_small) / scale).astype(np.int64)
        aligned_here = np.roll(im1, shift=shift_here, axis=(0, 1))

        aligned_final, shift_rest = pyramid_search(aligned_here, im2, metric, scale * 2, window=3)
        total_shift = (shift_here[0] + shift_rest[0], shift_here[1] + shift_rest[1])
        return aligned_final, total_shift

def analyze_single_scale():
    jpgs = ['cathedral.jpg', 'monastery.jpg', 'tobolsk.jpg']
    
    for imname in jpgs:
        im = cv2.imread(f"data/{imname}", cv2.IMREAD_GRAYSCALE) # (1024, 390) uint8
            
        # separate color channels
        height = np.floor(im.shape[0] / 3.0).astype(np.int64)
        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        b = crop(b)
        g = crop(g)
        r = crop(r)

        metric = "L2"
        ag, ag_shift = align(g, b, metric)
        ar, ar_shift = align(r, b, metric)
        im_out = np.dstack([b, ag, ar])

        # save the image
        print(f"Green channel shift (x,y): {(int(ag_shift[1]), int(ag_shift[0]))}, Red channel shift (x,y): {(int(ar_shift[1]), int(ar_shift[0]))}")
        fname = f'./out/{imname.split(".")[0]}_singlescale_{metric}.jpg'
        cv2.imwrite(fname, im_out)
        show_fit('Aligned Image', im_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

CUSTOM_IMAGES = ['beans.jpg', 'ceramic.jpg', 'woman.jpg']

def analyze_multi_scale():
    imgs = glob.glob('data/*.jpg') + glob.glob('data/*.tif')
    imgs = [f for f in imgs if os.path.basename(f) not in CUSTOM_IMAGES]
    for imname in imgs:
        im = cv2.imread(imname, cv2.IMREAD_GRAYSCALE) # (1024, 390) uint8

        # separate color channels
        height = np.floor(im.shape[0] / 3.0).astype(np.int64)
        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        b = crop(b)
        g = crop(g)
        r = crop(r)

        metric = "L2"
        ag, ag_shift = pyramid_search(g, b, metric)
        ar, ar_shift = pyramid_search(r, b, metric)
        im_out = np.dstack([b, ag, ar])

        # save the image
        print(f"Green channel shift (x,y): {(int(ag_shift[1]), int(ag_shift[0]))}, Red channel shift (x,y): {(int(ar_shift[1]), int(ar_shift[0]))}")
        basename = os.path.splitext(os.path.basename(imname))[0]
        fname = f'./out/{basename}_multiscale_{metric}.jpg'
        cv2.imwrite(fname, im_out)
        show_fit('Aligned Image', im_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def analyze_custom_images():
    for imname in CUSTOM_IMAGES:
        im = cv2.imread(f"data/{imname}", cv2.IMREAD_GRAYSCALE)

        # separate color channels
        height = np.floor(im.shape[0] / 3.0).astype(np.int64)
        b = im[:height]
        g = im[height: 2*height]
        r = im[2*height: 3*height]

        b = crop(b)
        g = crop(g)
        r = crop(r)

        metric = "L2"
        ag, ag_shift = pyramid_search(g, b, metric)
        ar, ar_shift = pyramid_search(r, b, metric)
        im_out = np.dstack([b, ag, ar])

        # save the image
        print(f"Green channel shift (x,y): {(int(ag_shift[1]), int(ag_shift[0]))}, Red channel shift (x,y): {(int(ar_shift[1]), int(ar_shift[0]))}")
        basename = os.path.splitext(imname)[0]
        fname = f'./out/{basename}_multiscale_{metric}.jpg'
        cv2.imwrite(fname, im_out)
        show_fit('Aligned Image', im_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ===== MAIN ====== #
def __main__():
    print("Analyzing single scale images...")
    analyze_single_scale()
    print("Analyzing multi-scale images...")
    analyze_multi_scale()
    print("Analyzing custom images...")
    analyze_custom_images()

if __name__ == "__main__":
    __main__()