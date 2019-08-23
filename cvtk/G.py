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


def G_estimator(cov, total_var, average_replicates=False, abs=False):
    assert(cov.ndim == 3)
    assert(total_var.ndim == 2)
    assert(total_var.shape[0] == cov.shape[0])
    T, T_, R = cov.shape
    Gs = list() # for cumulative G
    for t in np.arange(1, T+1):
        if average_replicates:
            G_reps = calc_G(np.nanmean(cov, axis=2), 
                            np.nanmean(total_var, axis=1),
                            end=t, abs=abs)
        else:
            G_reps = list()
            for r in np.arange(R):
                G = calc_G(cov[:, :, r], total_var=total_var[:, r], 
                            end=t, abs=abs)
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
             
