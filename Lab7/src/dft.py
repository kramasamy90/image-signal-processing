import numpy as np
from scipy.fft import fft, ifft

def fft_1d(U):
    '''
    This is not exactly 1D-FFT. Simillar to standard fft from scipy.
    Just takes care of the dimension.
    It returns \phi x U
    Args:
        U -> (np.ndarray): m x n dimension array.
    
    Returns:
        F -> Returns 1D-FFT of U.
    '''

    return fft(U.T).T

def ifft_1d(F):
    '''
    Inverse of fft_1d.
    It returns \phi* x F
    Args:
        img -> (np.ndarray): m x n dimension array.
    
    Returns:
        F -> Returns 1D-FFT of img. 
    '''

    return ifft(F.T).T

def fft_2d(img):
    '''
    Given input U get 2D FFT of U.
    Args:
        img -> (np.ndarray): m x n x 3
    '''
    img2d = np.mean(img, axis=2)
    return fft_1d(fft_1d(img2d.T).T)


def ifft_2d(F):
    img_2d = ifft_1d(ifft_1d(F.T).T)
    img_2d = (img_2d / np.max(img_2d)) *  255
    img_2d = img_2d.astype('int')
    # return img_2d
    img = np.zeros((img_2d.shape[0], img_2d.shape[1], 3), dtype='int')
    for i in range(3):
        img[:, :, i] = img_2d
    return img
    