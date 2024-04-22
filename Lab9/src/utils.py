import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def plot_rgb(ax, image, title):

    # Reshape the image to a 1D array of RGB values
    rgb_values = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rgb_values.append(image[i, j])
    rgb_values = np.array(rgb_values)

    # Extract R, G, B values
    r = rgb_values[:, 0]
    g = rgb_values[:, 1]
    b = rgb_values[:, 2]

    ax.scatter(r, g, b, c=rgb_values / 255.0, marker='o')

    # Set axis labels
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')

    ax.set_xlim(0, 255) 
    ax.set_ylim(0, 255) 
    ax.set_zlim(0, 255) 

    ax.set_title(title)
    return ax
