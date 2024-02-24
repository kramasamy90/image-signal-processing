import numpy as np
from tqdm import tqdm
from BWImage import BWImage
from utils import to_cartesian
from utils import to_homogenous

def stitch(canvas, imgs, _H, tx, ty):
    '''
    Stitch images in imgs onto a canvas.
    Args:
        canvas (BWImage): Canvas onto which the stitched images are written.
        imgs (list of BWImages): [Img1, Img2, ..., Img-n].
        H (list of np.arrays): List of homographies [H12, H23, ..., H(n-1, n)].
        tx, ty (int): Translation of Img2 w.r.t to the canvas origin.
    '''
    # Get H, where H[i] = _H[i] x _H[i-1] x ... x I.
    # NOTE: Reuse of H in a different meaning here compared to notebook.
    I = np.eye(3)
    H = [I]
    for i in range(len(_H)):
        H.append(_H[i] @ H[-1])
    h, l = canvas.shape()
    for i in tqdm(range(h)):
        for j in range(l):
            v = 0
            num_vals = 0
            sum_vals = 0
            for k in range(len(H)):
                _ = to_cartesian(H[k] @ to_homogenous(np.array([i - tx, j - ty]).reshape(2, 1)))
                x, y = _[0][0], _[1][0]
                v = imgs[k][x, y]
                if v < 0:
                    continue
                else:
                    num_vals += 1
                    sum_vals += v
                
            if num_vals > 0:            
                canvas[i, j] = sum_vals / num_vals

    return canvas