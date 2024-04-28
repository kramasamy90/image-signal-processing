import numpy as np


def get_mse(img1, img2, w):
    '''
    Calculate mean square distance between the images.
    '''


    pass


def get_psnr(_img1, _img2, w):
    '''
    Return Peak signal-to-noise ratio.
    Args:
        img1 (np.nadarray)            -> Image with no noise.
        img2 (np.nadarray)            -> Image with noise.
        w          (int)              -> Exclude the boundary of width 'w'.
    
    Returns:
        psnr (float)                  -> Returns the PSNR.
    '''
    m, n, _ = _img1.shape
    img1 = _img1[w:m-w, w:n-w] / 255
    img2 = _img2[w:m-w, w:n-w] / 255
    mse = (np.linalg.norm(img1 - img2) ** 2) / ((m - 2 * w) * (n - 2 * w))
    return 10 * np.log10(1 / mse)