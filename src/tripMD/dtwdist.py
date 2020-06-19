from dtaidistance import dtw_ndim
import numpy as np


def compute_ndim_dtw_dist_mat(ts_list, max_dist=None):
    """
    This function computes the pairwise distance matrix of a list of multidimensional time-series with Dynamic Time
    Warping distance. It is based on dtaidistance package

    :param ts_list: list of multidimensional time-series to compare pairwise. This is a list is lists, where each list
    correspond to a time-series and includes 1-D numpy arrays. Each numpy array is a single multidimensional observation
    of that time-series. Thus, the size of the numpy arrays must be all the same for all the trips and they encode the
    feature dimensions.
    :type ts_list: list of lists of 1D array
    :param max_dist: distance upper bound - if distance is higher than max_dist, then computation stops and the distance
     is set as inf this parameter serves merely for speeding up computation
    :type max_dist: float
    :return: distance matrix
    :rtype: 2D array
    """
    dist_matrix_vec = dtw_ndim.distance_matrix(
        ts_list, parallel=True, max_dist=max_dist
    )
    dist_matrix = np.triu(dist_matrix_vec) + np.triu(dist_matrix_vec).T
    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix


def compute_ndim_dwt_dist_between_ts_and_list(single_ts, ts_list, max_dist=None):
    dist_list = []
    for ts in ts_list:
        dist = dtw_ndim.distance(single_ts, ts, max_dist=max_dist)
        dist_list.append(dist)
    return dist_list
