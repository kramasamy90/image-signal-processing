from tqdm import tqdm

import numpy as np
from scipy.fft import fft, ifft

def fft_1d(U):
    '''
    Calculates DFT in column-wise manner.

    Args:
        U (np.ndarray) -> Dimension:m x n.
    
    Returns:
        (np.ndarray) -> Dimension:m x n. Returns 1D-DFT of U. 
        Here each column of F is the DFT of each column of U.
    '''

    return fft(U.T).T

def ifft_1d(F):
    '''
    Calculates inverse DFT in column-wise manner.

    Args:
        F (np.ndarray) -> m x n dimension array.
    
    Returns:
        U (np.ndarray) -> m x n dimension array. Returns inverse DFT of F.
        Here each column of U is the inverse DFT of each column of F.
    '''

    return ifft(F.T).T

def fft_2d(img):
    '''
    Given input image img calculate 2D FFT of img.

    Args:
        img (np.ndarray) -> m x n x 3.

    Returns:
        F (np.ndarray) -> Dimension: m x n. 2D-DFT of img.
    '''

    img2d = np.mean(img, axis=2)
    return fft_1d(fft_1d(img2d.T).T)


def ifft_2d(F):
    '''
    Given input image img calculate 2D FFT of img.

    Args:
        img (np.ndarray) -> m x n x 3.

    Returns:
        F (np.ndarray) -> Dimension: m x n. 2D-DFT of img.
    '''
    # Calculate inverse DFT.
    img_2d = ifft_1d(ifft_1d(F.T).T)
    img_2d = img_2d.astype('int')

    # Add 3 channels.
    img = np.zeros((img_2d.shape[0], img_2d.shape[1], 3), dtype='int')
    for i in range(3):
        img[:, :, i] = img_2d
    return img
    

def dft_rotate(img, theta):
    '''
    Returns DFT where the rotation is applied between the indices.
    CAUTION: Works only for square images.

    Args:
        img (np.ndarray) -> Dimension: m x n x 3.
        theta (float) -> Angle of rotation.
    
    Returns:
        F (np.ndarray) -> Dimension:m x n. Rotated DFT of img.
    '''

    N = img.shape[0]
    img2d = np.mean(img, axis=2)
    print(img2d.shape)
    F = np.zeros((N, N), dtype='complex')
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -1 * s], [s, c]])

    for k in tqdm(range(N)):
        for l in range(N):
            for m in range(N):
                for n in range(N):
                    vm = np.array([m, n]).reshape((2, 1))
                    vk = np.array([k, l]).reshape((2, 1))
                    e = -1j *(2 * np.pi / N) * (vm.T @ R @ vk)[0, 0]
                    F[k, l] += img2d[m, n] * np.exp(e)
    
    return F 