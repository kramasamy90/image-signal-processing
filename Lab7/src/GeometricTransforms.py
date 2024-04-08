import numpy as np

def rotate_translate(source, target, R, T):
    '''
    Rotate and translate the source to get target.

    Usage
        rotate_translate(source, target, R, T)
    
    Args:
        source -> (BWImage): Source image.
        target -> (BWImage): Target image.
        R -> (numpy array of dimension 2x2)
            A rotation matrix (in the context of Lab-2)
        T -> (numpy array of dimension 2x1)
            Translation vector.
    
    What it does:
        For each i, j in target, obtain the source pixel.
        [i, j]_{target} = R * [i, j]_{source} + T.
        [i, j]_{source} = R^T * ([i, j]_{target} - T) 
        target[[i, j]target] = source[[i, j]_source]
    '''
    xdim, ydim = target.shape()
    for i in range(xdim):
        for j in range(ydim):
            p = np.array([i, j])
            p = p.reshape(2, 1)
            # _i, _j = corresponding coordinates in source.
            # Obtain the values by translation and rotation.
            _p = R.T @ (p - T)
            _i, _j = _p
            # NOTE: source[_i, _j] return values based on bilinear interpolation
            # when _i or _j is not an integer.
            target[i, j] = source[_i, _j] 
