import numpy as np
from tqdm import tqdm

def get_nearest_cluster(x, clusters):
    '''
    Get the nearest cluster center to x.

    Args:
        x  (numpy.ndarray)   -> Dim: d x 1, input point.
        cs (numpy.ndarray)   -> Dim: d x k, k cluster centers.
    
    Returns: [nearest_cluster_id, nearest_dist]
        nearest_cluster_id (int)           -> Index of the closest center.
        nearest_dist       (float)         -> Distance between x and c. 
    '''

    d, k = clusters.shape
    nearest_cluster = clusters[:, 0]
    nearest_dist = np.linalg.norm(nearest_cluster - x)
    id = 0
    nearest_cluster_id = id
    while id < k-1:
        id += 1
        cluster = clusters[:, id]
        dist = np.linalg.norm(cluster - x)
        if dist < nearest_dist:
            nearest_cluster_id = id
            nearest_dist = dist
            nearest_cluster = cluster
    
    return [nearest_cluster_id, nearest_dist]

def kmeans(x, k, clusters = None, max_iters = None, print_interval = 10):

    '''
    Perform k means clustering on input data x.

    Args:
        x (np.ndarray)                   -> Input array. Dimension d x N.
                                             N - Number of datapoints.
                                             d - dimension of each datapoint.
        
        k                                -> Number of clusters.
        clusters                         -> Initial cluster centers.
        num_iters                        -> Limit on number of iterations.
    
    Returns: [cluster_assignment, clusters, cost]
        cluster_assignment (np.ndarray)  -> Cluster assignment. 1D array of length N.
        clusters                         -> Cluster centers. Dim: k x N.
        cost                             -> Cost corresponding to the cluster assignment.
    '''

    d, N = x.shape

    # 1. Obtain random initial k centers and assign points to clusters.
    min_x = np.min(x, axis=1)
    max_x = np.max(x, axis=1)
    if clusters is None:
        clusters = []
        for i in range(d):
            clusters.append(np.random.uniform(min_x[i], max_x[i], k))
        clusters = np.array(clusters)
    cluster_assignment = np.zeros(N, dtype=int)

    iter = 0
    while(True):
        if iter == max_iters:
            break
        iter += 1

        cluster_id_is_same = True
        cluster_size = np.zeros(k)
        cost = 0

    # 2. Assign points to nearest clusters.
        for i in range(N):
            id, dist = get_nearest_cluster(x[:, i].reshape(d, 1), clusters)
            cluster_id_is_same = cluster_id_is_same and (cluster_assignment[i] == id)
            cluster_assignment[i] = id
            cluster_size[id] += 1
            cost += dist ** 2

    # 3. Recalculate cluster center.

        clusters = np.zeros(d * k, dtype=np.float64).reshape(d, k)
        for i in range(N):
            clusters[:, cluster_assignment[i]] += x[:, i]
        
        for i in range(k):
            if cluster_size[i] == 0:
                continue
            clusters[:, i] = clusters[:, i] / cluster_size[i]

    # 4. Break upon convergence.
        if cluster_id_is_same:
            print("Convergence")
            break
    
        if (iter % print_interval == 0):
            print(f'Iteration: {iter}, Cost: {cost}')

    return [cluster_assignment, clusters, cost]
    
