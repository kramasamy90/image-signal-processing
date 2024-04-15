import numpy as np

def get_pixel_array(img):
    '''
    Given a RGB image return and array of pixels.

    Args:
        img (np.ndarray) -> Dim: m x n x 3. 
    
    Returns:
        pixel_array (np.ndarray) -> Dime: 3 x mn. Each column is a pixel.
                                    k-th column is the pixel (k // n, k % n)
    '''
    m, n, _ = img.shape
    pixel_array = np.zeros(m * n * 3).reshape(3, m*n)

    for i in range(m):
        for j in range(n):
            pixel_array[:, i * n + j] = img[i, j]
    
    return pixel_array

def get_image(pixel_array, m, n):
    '''
    Given a pixel_array return an RGB image.

    Args:
        pixel_array (np.ndarray) -> Dime: 3 x mn. Each column is a pixel.
                                    k-th column is the pixel (k // n, k % n)
        m (int)                  -> Number of rows.
        n (int)                  -> Number of columns.

    Returns:
        img (np.ndarray) -> Dim: m x n x 3. 
    '''

    img = np.zeros((m, n, 3), dtype=int)

    for i in range(m):
        for j in range(n):
            img[i, j] = pixel_array[:, i * n + j]
    
    return img