import numpy as np
from cvtk.bootstrap import weighted_mean
from cvtk.cov import stack_replicate_covariances, stack_temporal_covariances
from cvtk.cov import stack_replicate_covs_by_group, stack_temporal_covs_by_group
from cvtk.cov import temporal_replicate_cov

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


def convergence_corr_by_group(covs, R, T):
     replicate_covs = stack_replicate_covs_by_group(covs, R, T, upper_only=False)
     temporal_covs = stack_temporal_covs_by_group(covs, R, T)
     num, denom = (convergence_corr_numerator(replicate_covs),
                   convergence_corr_denominator(temporal_covs))
     return num / denom 

def convergence_corr_from_freqs(freqs, depths=None, diploids=None, **kwargs):
     R, Tp1, L = freqs.shape
     T = Tp1 - 1
     covs = temporal_replicate_cov(freqs, depths=depths, diploids=diploids, **kwargs)
     return convergence_corr(covs, R, T)

def convergence_corr(covs, R, T):
     replicate_covs = stack_replicate_covariances(covs, R, T, upper_only=False)
     temporal_covs = stack_temporal_covariances(covs, R, T)
     num, denom = (convergence_corr_numerator(replicate_covs[np.newaxis, ...]),
                   convergence_corr_denominator(temporal_covs[np.newaxis, ...]))
     return num / denom 

def convergence_corr_numerator(stacked_replicate_covs):
    """
    E_{A≠B} cov(Δp_{t,A}, Δp_{s,B})
    """
    assert(stacked_replicate_covs.ndim == 4)
    #return stacked_replicate_covs
    return np.mean(stacked_replicate_covs, axis=3)

def convergence_corr_denominator(stacked_temporal_covs):
    """
    E_{A≠B} sqrt(var(Δp_{t,A}) var(Δp_{s,B}))
    """
    assert(stacked_temporal_covs.ndim == 4)
    nblocks, T, T_, R = stacked_temporal_covs.shape
    vars = np.diagonal(stacked_temporal_covs, offset=0, axis1=1, axis2=2)
    # get the outer products over time and replicates
    varmat = np.einsum('bri,bqj->brqij', vars, vars)
    # the following gets all A, B replicate pairs where A ≠ B
    tr, tc = np.triu_indices(R, k=1)
    sdmat = np.sqrt(varmat[:, tr, tc, :, :].mean(axis=1))
    return sdmat
    
 
