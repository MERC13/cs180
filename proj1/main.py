import numpy as np
import cv2
import glob

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

def align(im1, im2, metric, window=7):
    best_shift = (0, 0)
    best_score = float('inf') if metric == "L2" else -float('inf')
    shift_range = range(-window, window + 1)

    for dx in shift_range:
        for dy in shift_range:
            shifted_im1 = np.roll(im1, shift=(dy, dx), axis=(0, 1))

            if metric == "L2":
                score = np.sqrt(np.sum((shifted_im1.astype(np.float64) - im2) ** 2))
                if score < best_score:
                    best_score = score
                    best_shift = (dy, dx)
            elif metric == "NCC":
                mean_im1 = shifted_im1 - shifted_im1.mean()
                mean_im2 = im2 - im2.mean()
                score = np.sum(mean_im1 * mean_im2) / (np.linalg.norm(mean_im1) * np.linalg.norm(mean_im2))

                if score > best_score:
                    best_score = score
                    best_shift = (dy, dx)
        
    return np.roll(im1, shift=best_shift, axis=(0, 1)), best_shift

def pyramid_search(im1, im2, metric, scale=0.0625, window=15):
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
        fname = f'./out/{imname.split(".")[0]}_aligned_{metric}.jpg'
        cv2.imwrite(fname, im_out)
        show_fit('Aligned Image', im_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def analyze_multi_scale():
    imgs = glob.glob('data/*.jpg')+glob.glob('data/*.tif')
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
        fname = f'./out/{imname.split("/")[-1].split(".")[0]}_aligned_{metric}.jpg'
        cv2.imwrite(fname, im_out)
        show_fit('Aligned Image', im_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ===== MAIN ====== #
def __main__():
    print("Analyzing single scale images...\n\n\n")
    analyze_single_scale()
    print("Analyzing multi-scale images...\n\n\n")
    analyze_multi_scale()

if __name__ == "__main__":
    __main__()