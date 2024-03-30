import numpy as np
from BWImage import BWImage
from tqdm import tqdm

def get_ml(img):
    '''
        Gives modified laplacian for the input image.

        Args:
            img (numpy.ndarray) -> n x m numpy array.

        Returns:
            ml (numpy.ndarray) -> n-1 x m-1 numpy array
    '''

    n, m = img.shape
    ker = np.array([1, -2, 1], dtype=np.float64)
    ml = np.zeros((n-1, m-1), dtype=np.float64)

    for i in range(m-2):
        for j in range(n-2):
            fxx = 0 # d^2/dx^2 along x.
            fyy = 0 # d^2/dy^2 along y.
            for k in range(3):
                # NOTE: The target pixel is i+1, j+1.
                fxx += img[i + k][j+1] * ker[k]
                fyy += img[i+1][j + k] * ker[k]
            ml[i][j] = np.abs(fxx) + np.abs(fyy)
    
    return ml


def get_sml(img, q):
    '''
    Returns Sum-modified Laplacian.

        Args:
            img (numpy.ndarray) -> n x m numpy array.
            q (int) -> Window size for SML.

        Returns:
            sml (numpy.ndarray) -> n-1 x m-1 numpy array
    '''

    m, n = img.shape
    sml = np.zeros((m - 2 * q, n - 2 * q))

    ml = get_ml(img)
    for i in range(q, m - q):
        for j in range(q, n - q):
            sml[i-q, j-q] = np.sum(ml[i-q:i+q+1, j - q:j + q + 1])

    return sml

def get_depth(Fx, Fy, Fz, dx, dy, dz):
    '''
    Given the intensities at three depths Fm, F_m-1, F_m+1 return d_bar.
    Args:
        Fx, Fy, Fz. (int) -> Intensity values, F_m-1, F_m, F_m+1 respectively.
        dx, dy, dz. (int) -> Depth d_m-1, d_m, d_m+1 respectively.
    Returns:
        d (float) -> Depth with maximum intensity.
    '''
    dx = dx
    dy = dy
    dz = dz

    # Case where Fx, Fy and Fz are very close to each other. 
    if (2 * np.log(Fy) - np.log(Fx) - np.log(Fz) < 1e-12):
        return dy

    d_bar = (dz**2 - dy**2)*(np.log(Fy) - np.log(Fx)) \
            - (dx**2 - dy**2)*(np.log(Fy) - np.log(Fz))
    d_bar = d_bar / (2 * (2 * np.log(Fy) - np.log(Fx) - np.log(Fz)))

    return d_bar

def get_sff(img_stack, q, dof = 50.5):
    '''
    Given stack of images returns the depth at each pixel.

    Args:
        img_stack (numpy.ndarray) -> A numpy array of dimension K x M x N.
        q (int) -> Window size for SML.

    Returns:
        img_depth -> A numpy array of dimension (M - 2q) x (N - 2q).
    '''
    # Get sum-modified laplacian for each image in the image stack.
    K, M, N = img_stack.shape
    sml_stack = []
    sff = np.zeros((M - 2*q, N - 2*q))

    for img in img_stack:
        sml_stack.append(get_sml(img, q))

    for i in range(M - 2 * q):
        for j in range(N - 2 * q):
            v = np.array([sml_stack[k][i][j] for k in range(K)])
            v = v + 1e-6 # To avoid log(0) error in get_d function.
            m = np.argmax(v)
            if m == 0:
                sff[i][j] = 0
            elif m == K-1:
                sff[i][j] = (K-1) * dof
            else:
                sff[i][j] = get_depth(v[m-1], v[m], v[m+1], m-1, m, m+1) * dof

    return sff