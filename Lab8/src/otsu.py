import numpy as np

def get_histogram(img, channel = 0, depth = 8):
    '''
    Get the histgram of intesity for input image for a channel.
    Args:
        img (np.ndarray) -> m x n x 3.
        channel (int) -> Gets histogram for this channel.
        depth (int) -> Bit depth of the image.
    
    Returns:
        h (np.ndarray) -> Numpy array with frequency at each intensity.
    '''
    H = np.zeros(2 ** depth)
    m, n, _ = img.shape
    for i in range(m):
        for j in range(n):
            H[int(img[i, j, channel])] += 1
    
    return H

def get_between_class_variance(sum_l, sum_r, n_l, n_r):
    '''
    Get between_class_variance.
    Args:
        sum_l (int) -> Sum of intentsities of left class.
        sum_r (int) -> Sum of intentsities of right class.
        n_l   (int) -> Sum of frequencies of left class.
        n_r   (int) -> Sum of frequencies of right class.
    
    Returns:
        between_dev (float) -> Between class variance.
    '''
    m1 = sum_l / n_l
    m2 = sum_r / n_r
    m = (sum_l + sum_r) / (n_l + n_r)
    between_dev = (((m1 - m)**2) * (n_l / (n_l + n_r))) + (((m2 - m)**2) * (n_r / (n_l + n_r)))
    return between_dev

def get_between_class_variance_recursively(sum_l, sum_r, n_l, n_r, hist, t):
    '''
    Get variance across the two groups separated by threshold t.
    It exploits the previous calculation of sum.

    Args:
        sum_l (int)       -> Sum till t-1. [Previous calculation of sum]
        sum_r (int)       -> Sum from t. [Previous calculation of sum]
        n_l (int)         -> Number of pixels till t-1.
        n_r (int)         -> Number of pixels from t.
        hist (np.ndarray) -> Histogram.
    
    Returns:

    '''
    sum_l += t * hist[t]
    sum_r -= t * hist[t]
    n_l += hist[t]
    n_r -= hist[t]
    m1 = sum_l / n_l
    m2 = sum_r / n_r
    m = (sum_l + sum_r) / (n_l + n_r)
    between_dev = (((m1 - m)**2) * (n_l / (n_l + n_r))) + (((m2 - m)**2) * (n_r / (n_l + n_r)))
    return (sum_l, sum_r, n_l, n_r, between_dev)

def otsu(hist):
    '''
    Return threshold calculated based on Otsu's method

    Args:
        hist (np.ndarray) -> 1D array of pixel intensity histogram.
    
    Returns:
        t (int) -> Threshold calculated based on Otsu's method.

    '''
    sum_all = np.sum(np.array([i * hist[i] for i in range(hist.shape[0])]))
    n = np.sum(hist)

    l_lim = 0
    while(hist[l_lim] == 0):
        l += 1
    r_lim = hist.shape[0] - 1
    while(hist[r_lim] == 0):
        r_lim -= 1
    
    between_devs = []
    for i in range(l_lim):
        between_devs.append(0)

    sum_l = l_lim * hist[l_lim]
    n_l = hist[l_lim]

    sum_r = sum_all - sum_l
    n_r = np.sum(hist[l_lim+1:])

    max_dev = 0
    max_t = 0

    between_devs.append(get_between_class_variance(sum_l, sum_r, n_l, n_r))

    for i in range(l_lim + 1, r_lim):
        sum_l, sum_r, n_l, n_r, between_dev = get_between_class_variance_recursively(sum_l, sum_r, n_l, n_r, hist, i + 1)
        between_devs.append(between_dev)
        if between_dev > max_dev:
            max_dev = between_dev
            max_t = i + 1

    return max_t, between_devs
