import numpy as np
from cvtk.bootstrap import weighted_mean

def calc_G(cov, total_var, end=None, abs=False):
    """
    Params:
          - cov: a temporal covariance matrix
          - end: the last timepoint to include of the covariance matrix
          - abs: take absolute value
          - ignore_adjacent: ignore adjacent timepoints, the k = +/- 1 offdiagonal
               (these undergo bias correction).
                
    """
    assert(cov.ndim == 2)
    assert(total_var.shape[0] == cov.shape[0])  # same number of timepoints
    assert(cov.shape[0] == cov.shape[1])
    T = cov.shape[0]
    end = T+1 if end is None else end
    k = 1
    offdiag = np.tril(cov, -k) + np.triu(cov, k)
    #import pdb; pdb.set_trace()
    if abs:
        offdiag = np.abs(offdiag[:end, :end])
    total_cov = np.nansum(offdiag[:end, :end])
    total_var = total_var[end-1]  # TODO
    G = total_cov / total_var
    return G


def block_estimate_G(array, weights, total_var, all_timepoints=False, abs=False):
    assert(array.ndim == 4)
    nblocks, T, T_, R = array.shape
    mean_covs = weighted_mean(array, weights)
    mean_total_var = weighted_mean(total_var, weights)
    
    Gs = list() # for cumulative G
    for t in np.arange(T+1):
        G_reps = list()
        for r in np.arange(R):
            G = calc_G(mean_covs[:, :, r], total_var=mean_total_var[:, r], 
                        end=t, abs=abs)
            G_reps.append(G)
        Gs.append(G_reps)
    G_array = np.array(Gs)
    return G_array



def calc_G_deprecated(cov, end=None, abs=False, ignore_adjacent=False):
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


def block_estimate_G_deprecated(array, weights, all_timepoints=False, abs=False, 
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


def replicate_cor_coef(array, weights, R, T):
    """
    E_{A≠B} cov(Δp_{t,A}, Δp_{t,B})
    -----------------------
    E_{A≠B} sqrt(var(Δp_{t,A}) var(Δp_{t,B}))
    """
    RxT, RxT = array.shape
    #rep_mats = stack_replicate_covs_by_group(array, R, T)
    #temp_mats = stack_temporal_covs_by_group(array, R, T)  
    rep_row, rep_col = replicate_block_matrix_indices(R, T)
    for A in np.arange(R):
        for B in np.arange(A, R):
            if A == B:
                continue
            (rep_row == A) & (rep_col == B)
             
