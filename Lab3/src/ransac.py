import numpy as np
from scipy.linalg import null_space
from utils import to_cartesian
from utils import to_homogenous

def homography_4pts(pts1, pts2):
    '''
    Calculate homography based on 4 point-correspondences.

    Args:
        pts1 (np.array): 4 points from image-1.
        pts2 (np.array): 4 points from image-2.
    
    Returns:
        H_21 (np.array): Homography such that I_1 = H_21 @ I_2
    '''

    A = np.zeros((8, 9), dtype='float64')
    for i in range(4):
        x_, y_ = pts1[i][0], pts1[i][1]
        x, y = pts2[i][0], pts2[i][1]
        A[2 * i] = np.array([-1 * x, -1 * y, -1, 0, 0, 0, x * x_, y * x_, x_])
        A[2 * i + 1] = np.array([0, 0, 0, -1 * x, -1 * y, -1, x * y_, y * y_, y_])
    _H = null_space(A)
    H = _H.reshape((3, 3))
    return H


def ransac(cpts):
    '''
    Use RANSAC algorithm to calculate homography 
    from point correspondences.

    Args:
        cpts (np.array): Dimension = 2 x n x 2.
    
    Returns:
        Homography (np.array): Dimension 3 x 3.
    '''

    ## Input and output variables.
    # Number of point correspondences.
    n = cpts.shape[1] 
    print("n = ", n)
    # Largest consensus set.
    largest_consensus_set = set()

    ## Parameters:
    # Error threshold (epsilon) to determine.
    err = 1
    # Good consensus set size.
    d = 0.8 *n
    print("d = ", d)
    # Maximum number of iterations.
    # !! Change num_iters.
    num_iters = 10

    ### Core RANSAC algorithm.
    for i in range(num_iters):
        print("i = ", i)
        ## Choose 4 random points.

        # I -> Indices of the chosen 4 random points
        I = np.random.choice(n, 4, replace=False).tolist()
        # Some set operations to get D.
        # D -> list of indices not in I.
        I_set = set(I)
        U_set = set([i for i in range(n)])
        D = list(U_set - I_set)
        
        ## Get homography.
        pts1 = cpts[0][I]
        pts2 = cpts[1][I]
        H = homography_4pts(pts1, pts2)

        # Get consensus set.
        consensus_set = set()
        for j in D:
            # i-th point from img1.
            pt1 = cpts[0][j].reshape(2,1)
            # i-th point from img2.
            pt2 = cpts[1][j].reshape(2,1)

            # Convert to homogenous coordinates.
            # Apply homography and  convert back to cartesian.
            pt1_ = to_cartesian(H @ to_homogenous(pt2))

            if((np.linalg.norm(pt1_ - pt1)) < err):
                # print("j = ", j)
                consensus_set.add(j)
            
        if (len(consensus_set) + 4 > len(largest_consensus_set)):
            # +4 to account for I.
            largest_consensus_set = consensus_set | I_set
        
        if (len(largest_consensus_set) >= d):
            print("Largest CS = ", len(largest_consensus_set))
            break
    
    print(len(largest_consensus_set))
    return H



# !! Clean-up testing code.
#====================================================
#    TESTING
#====================================================


def multH(pt2, H):
    pt_h2 = to_homogenous(pt2.reshape(2, 1))
    pt_h1 = H @ pt_h2
    return to_cartesian(pt_h1)

if (__name__ == '__main__'):
    H = np.array([1, 2, 1, 0, 10, 0, 0, 0, 1]).reshape(3, 3)

    cpts2 = [0, 0, 0, 1, 1, 0, 2, 2, 3, 10, 12, 17]
    k = int(len(cpts2) / 2) 
    cpts2 = np.array(cpts2).reshape(k, 2)
    cpts1 = np.zeros(cpts2.shape)
    for i in range(k):
        cpts1[i] = multH(cpts2[i], H).reshape(2,)
    cpts = np.zeros(2 * k * 2).reshape(2, k, 2)
    cpts[0] = cpts1
    cpts[1] = cpts2

    print("***********")
    print(H / np.linalg.norm(H))
    print("***********")
    H = ransac(cpts)
    print(H)
