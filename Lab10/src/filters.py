from tqdm import tqdm

import numpy as np

def nlm_filter(_img, sigma, W_sim, W):
    '''
    Applies non-local means filtering on input image.

    Args:
        img     (np.ndarray)      -> Dim: m x n x 3. Input RGB image.
        sigma   (float)           -> Std. Dev for NLM filter.
        w_sim   (int)             -> Simillarity neighbourhood radius.
        w       (int)             -> Search neighbourhood radius.

    Returns:
        img_filtered (np.ndarray) -> Filtered image.
    '''
    # Change the intensity values to range 0-1.
    # img = _img / np.max(img, axis=(0, 1))
    img = _img / 255 # 255 happens to be the MAX for all the channels.
    m, n, _ = img.shape
    # The boundary part that gets lost.
    b = W_sim + W  
    # Output image in range 0-1.
    _img_filtered = np.zeros(img.shape)

    # Apply NLM filter for each pixel.
    for i in range(b, m-b):
        for j in range(b, m-b):
            w = np.zeros((2*W + 1, 2 * W + 1))
            for k in range(i - W, i + W + 1):
                for l in range(j - W, j + W + 1):
                    # print(i, j, k, l)
                    Np = img[i - W_sim:i + W_sim + 1, j - W_sim:j + W_sim + 1]
                    Nq = img[k - W_sim:k + W_sim + 1, l - W_sim:l + W_sim + 1]
                    d = np.linalg.norm(Np - Nq) ** 2
                    w[k - (i - W), l - (j - W)] = np.exp(-1 * d / sigma **2)
            # Normalize w.
            w = w / np.sum(w)
            for k in range(i - W, i + W + 1):
                for l in range(j - W, j + W + 1):
                    _img_filtered[i, j] += img[k, l] * w[k - (i - W), l - (j - W)]

    # Get output image in the range 0-255.
    img_filtered = _img_filtered * 255
    img_filtered = img_filtered.astype('uint8')
    return img_filtered
                    

def gaussian_filter(img, sigma):
    '''
    Apply Gaussian filtering on the input image.

    Args:
        img     (np.ndarray)      -> Dim: m x n x 3. Input RGB image.
        sigma   (float)           -> Std. Dev for the Gaussian filter.

    Returns:
        img_filtered (np.ndarray) -> Filtered image.
    '''

    img_filtered = np.zeros(img.shape)
    m, n, _ = img.shape
    # 1 D Gaussian kernel
    gk = np.zeros(7)
    for i in range(7):
        gk[i] = np.exp(-1 * (3 - i)**2 / (2 * sigma**2))
    gk = gk.reshape(7, 1)
    gk = gk / np.sum(gk)

    for i in range(3, m - 3):
        for j in range(3, n - 3):
            patch0 = img[i-3:i+4, j-3:j+4, 0]
            patch1 = img[i-3:i+4, j-3:j+4, 1]
            patch2 = img[i-3:i+4, j-3:j+4, 2]
            img_filtered[i, j, 0] = (gk.T @ patch0 @ gk)[0, 0]
            img_filtered[i, j, 1] = (gk.T @ patch1 @ gk)[0, 0]
            img_filtered[i, j, 2] = (gk.T @ patch2 @ gk)[0, 0]
    
    img_filtered = img_filtered.astype('uint8')
    return img_filtered
    

def get_nlm_filter(_img, sigma, W_sim, W, i, j):
    '''
    Applies non-local means filtering on input image.

    Args:
        img     (np.ndarray)      -> Dim: m x n x 3. Input RGB image.
        sigma   (float)           -> Std. Dev for NLM filter.
        w_sim   (int)             -> Simillarity neighbourhood radius.
        w       (int)             -> Search neighbourhood radius.
        i, j    (int, int)        -> Position on the image.

    Returns:
        w (np.ndarray)            -> The kernel at (i, j).
    '''
    # Change the intensity values to range 0-1.
    # img = _img / np.max(img, axis=(0, 1))
    img = _img / 255 # 255 happens to be the MAX for all the channels.
    m, n, _ = img.shape
    # The boundary part that gets lost.
    b = W_sim + W
    # Output image in range 0-1.
    _img_filtered = np.zeros(img.shape)

    # Apply NLM filter for each pixel.
    w = np.zeros((2 * W + 1, 2 * W + 1))
    for k in range(i - W, i + W + 1):
        for l in range(j - W, j + W + 1):
            # print(i, j, k, l)
            Np = img[i - W_sim:i + W_sim + 1, j - W_sim:j + W_sim + 1]
            Nq = img[k - W_sim:k + W_sim + 1, l - W_sim:l + W_sim + 1]
            d = np.linalg.norm(Np - Nq) ** 2
            w[k - (i - W), l - (j - W)] = np.exp(-1 * d / sigma **2)
    # Normalize w.
    w = w / np.max(w)

    return w

def get_gaussian_filter(sigma):
    '''
    Return Gaussian filter with sigma value given in input.

    Args:
        sigma   (float)           -> Std. Dev for the Gaussian filter.
    
    Returns:
        gk  (np.ndarray)          -> Gaussian kernel of size 11 x 11 corresponding to sigma as input.
    '''

    # 1 D Gaussian kernel
    gk = np.zeros(11)
    for i in range(11):
        gk[i] = np.exp(-1 * (5 - i)**2 / (2 * sigma**2))
    gk = gk.reshape(11, 1)
    gk = gk / np.sum(gk)
    gk = gk @ gk.T
    gk = gk / np.max(gk)
    return gk
