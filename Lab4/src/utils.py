import numpy as np
from BWImage import BWImage
from tqdm import tqdm

def gauss2D(x, sigma):
    '''
    Returns value of 2D Gaussian function at, x, y for the given sigma.

    Args:
        x, y (float): Just the x and y values for 2D-Gaussian.
        sigma (float): Standard deviation for the 2D- Gaussian.
    
    Return:
        (float) : A value proportional to the value of the 
                    Gaussian function with sigma as std. dev.  at (x, y).
                    NOTE: This is not the exact value of the Gaussian.
                    Because I do normalization of the filter, it is not
                    necessary to get the exact value of the Gaussian.
    '''

    return np.exp(-1 * (x **2) / (2 * sigma ** 2))

def get_gaussian_kernel(sigma):
    '''
    Gives a component of the Gaussian kernel for the input sigma.

    Args:
        sigma (float): Standard deviation for the Gaussian kernel.
    
    Returns:
        gk (np.ndarray): Returns a 1D np array. One component of the two components of the 
                            Gaussian kernel, which is a seperable kernel.
    '''

    if sigma == 0.0:
        return np.array([1]).reshape(1, 1)

    dim = np.ceil(6 * sigma + 1)
    dim = int(dim)
    if dim % 2 == 0:
        dim += 1
    
    gk = np.zeros(dim)

    for i in range(dim):
            gk[i] = gauss2D(i - (dim // 2), sigma)
    
    gk = gk / np.sum(gk)

    return gk.reshape(dim, 1)

def convolve(bwimg, kernel):
    '''
    Return img convolved with kernel.

    Args:
        bwimg (BWImage): Input image.
        kernel (np.ndarray): Kernel as 2D-Numpy array.

    Returns:
        (BWImage): Output after convolution with the kernel. 
    '''

    # Convolve the image with the original kernel
    m, n = bwimg.shape()
    l = kernel.shape[0]

    bwimg1 = BWImage()
    # bwimg1.make_blank((m - l + 1, n))
    bwimg1.make_blank(bwimg.shape())

    for i in tqdm(range(l // 2, m - (l // 2))):
         for j in range(n):
            value = 0.0
            for k in range(l):
                value += bwimg[i + k - (l // 2), j] * kernel[k, 0]
            bwimg1[i - (l // 2), j] = np.ceil(value)
    
    bwimg2 = BWImage()
    # bwimg2.make_blank((m - l + 1, n - l + 1))
    bwimg2.make_blank(bwimg.shape())

    for i in tqdm(range(l // 2, m - (l // 2))):
        for j in range(l // 2, m - (l // 2)):
            value = 0.0
            for k in range(l):
                value += bwimg1[i - (l // 2), j - (l // 2) + k] * kernel[k, 0]
            bwimg2[i - l // 2, j - l // 2] = value

    return bwimg2


    # Get new kernel by taking transpose of the original kernel
    # and convolve.