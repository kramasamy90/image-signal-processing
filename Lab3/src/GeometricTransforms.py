import numpy as np

def translate(source, target, tx, ty):
    '''
    Translate the image by tx and ty.

    Usage:
        translate(source, target, tx, ty)

    Args:
        source -> (BWImage): Source image.
        target -> (BWImage): Target image.
        tx (float): Translation along x.
        ty (float): Translation along y.

    Returns:
        (BWImage) A Translated image.
    '''

    xdim, ydim = target.shape()
    for i in range(xdim):
        for j in range(ydim):
            target[i, j] = source[i - tx, j - ty] # Note source[x, y] returns bilinear interpolation.


def rotate(source, target, theta):
    '''
    Rotate an image counter-clockwise by theta degrees.

    Usage example:
        rotate(source, target, theta)

    Args:
        source -> (BWImage): Source image.
        target -> (BWImage): Target image.
        theta -> (float): Angle in degrees.
    
    Returns:
        (BWImage) Source rotated image.
    '''

    # Convert theta to radians.
    theta = theta / 180 * np.pi
    # Get rotation matrix R.
    R = np.array([[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    xdim, ydim = target.shape()

    for i in range(xdim):
        for j in range(ydim):
            x, y = R.T @ np.array([i, j])
            target[i, j] = source[x, y] # Note source[x, y] returns bilinear interpolation.

def scale(source, target, x_scale, y_scale):
    '''
    Scale an image.

    Usage example:
        scale(source, target, theta)

    Args:
        source -> (BWImage): Source image.
        target -> (BWImage): Target image.
        x_scale -> Scale factor along x-axis.
        y_scale -> Scale factor along y-axis.
    
    Returns:
        (BWImage) Source, scaled image.
    '''

    xdim, ydim = target.shape()
    for i in range(xdim):
        for j in range(ydim):
            target[i, j] = int(source[i/x_scale, j/y_scale]) # Note source[x, y] returns bilinear interpolation.