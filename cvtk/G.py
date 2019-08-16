import numpy as np
from cvtk.bootstrap import weighted_mean

def calc_G(cov, end=None, abs=False, ignore_adjacent=False):
    """
    Params:
          - cov: a temporal covariance matrix
          - end: the last timepoint to include of the covariance matrix
          - abs: take absolute value
          - ignore_adjacent: ignore adjacent timepoints, the k = +/- 1 offdiagonal
               (these undergo bias correction).
                
    """
    assert(cov.ndim == 2)
    assert(cov.shape[0] == cov.shape[1])
    T = cov.shape[0]
    end = T+1 if end is None else end
    k = 1 if not ignore_adjacent else 2
    # extract the off diagonal elements, starting at off-diagonal k
    # if ignore_adjacent is true, this starts at the off, off diagonal
    # which doesn't share sampling noise that affects the covariance.
    offdiag = np.tril(cov, -k) + np.triu(cov, k)
    if abs:
        offdiag = np.abs(offdiag[:end, :end])
    covs = np.nansum(offdiag[:end, :end])
    double_offdiag = False  # experimental!
    if double_offdiag:
        offcovs = np.nansum(np.diag(cov[:end, :end], k=1))
        covs += offcovs
    total_var = np.nansum(cov[:end, :end])
    G = covs/total_var
    return G


def block_estimate_G(array, weights, all_timepoints=False, abs=False, 
                     ignore_adjacent=False):
    assert(array.ndim == 4)
    nblocks, T, T_, R = array.shape
    mean_covs =weighted_mean(array, weights)
    
    Gs = list() # for cumulative G
    for t in np.arange(T+1):
        G_reps = list()
        for r in np.arange(R):
            G = calc_G(mean_covs[:, :, r], end=t, abs=abs, 
                       ignore_adjacent=ignore_adjacent)
            G_reps.append(G)
        Gs.append(G_reps)
    G_array = np.array(Gs)
    return G_array


