import numpy as np

def to_cartesian(pt_h):
    '''
    Convert pt_h to cartesian  coordinate.
    Args:
        pt_h (np.array): shape = (k+1,1). A point in k-dimension homogenous coordinate.
    
    Returns:
        pt (np.array): shape = (k,1). k-dimension cartesian coordinate of pt_h. 
    '''
    k = pt_h.shape[0]
    pt = np.zeros(k-1).reshape(k-1, 1)
    for i in range(k-1):
        pt[i] = pt_h[i] / pt_h[k-1]
    
    return pt

def to_homogenous(pt):
    '''
    Convert pt to homogenous coordinate.

    Args:
        pt (np.array): shape = (k, 1). A k-dimension point in cartesian coordinate.
    
    Returns:
        pt_h (np.array): shape (k+1, 1). A (k)-dimension point in homogenouse coordinate.
    '''

    k = pt.shape[0]
    pt_h = np.zeros(k+1).reshape(k+1, 1)
    pt_h[k] = 1
    for i in range(k):
        pt_h[i] = pt[i]
    
    return pt_h
