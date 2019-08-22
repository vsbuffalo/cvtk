import numpy as np
from tqdm import tnrange
from cvtk.utils import view_along_axis


def bootstrap_ci(estimate, straps, alpha=0.05, method='pivot', stack=True):
    """
    Return pivot CIs
      This confidence interval returned is a pivotal CIs,
          C_l = 2 T - Q(1-α/2)
          C_u = 2 T - Q(α/2)
       where T is the estimator for the stastistic T, and α is the confidence level,
       and Q(x) is the empirical x percentile across the bootstraps.
    """
    alpha = 100. * alpha  # because, numpy.
    qlower, qupper = np.nanpercentile(straps, alpha/2, axis=0), np.nanpercentile(straps, 100-alpha/2, axis=0)
    if method == 'percentile':
        CIs = qlower, estimate, qupper
    elif method == 'pivot':
        CIs = 2*estimate - qupper, estimate, 2*estimate - qlower
    else:
        raise ValueError("method must be either 'pivot' or 'percentile'")
    if stack:
        return np.stack(CIs)
    return CIs


def weighted_mean(array, weights, axis=0):
    """
    Weighted mean for a block of resampled temporal covariance matrices. 
    This uses masked_array since nothing in numpy can handle ignoring nans and 
    weighted averages.
    """
    # mask the covariance matrix, since ma.average is the only nanmean
    # in numpy that takes weights
    array_masked = np.ma.masked_invalid(array)
    return np.ma.average(array_masked, axis=axis, weights=weights).data

def block_bootstrap(freqs, block_indices, block_seqids, B, 
                    estimator, depths=None, diploids=None, alpha=0.05, 
                    keep_seqids=None, return_straps=False,
                    ci_method='pivot', progress_bar=False, **kwargs):
    if progress_bar:
        B_range = tnrange(int(B), desc="bootstraps")
    else:
        B_range = range(int(B))

    # create a vector of the block indices to resample, based on 
    # whether keep_seqids is specified
    if keep_seqids is not None:
        blocks = np.array([i for i, seqid in enumerate(block_seqids) 
                           if seqid in keep_seqids], dtype='uint32')
    else:
        blocks = np.array([i for i, seqid in enumerate(block_seqids)], dtype='uint32')
    # number of samples in resample
    nblocks = len(blocks)
    straps = list()
    strap_lens = list()
    
    # build an array that's length of loci, full of the block IDs
    #block_ids = np.full(freqs.shape[2], -1, dtype='uint32')
    #for i, indices in enumerate(block_indices):
    #    block_ids[indices] = i
    #block_indices = np.array(block_indices)

    group_freqs = list()
    group_depths = None if depths is None else []
    
    for indices in block_indices:
        group_freqs.append(view_along_axis(freqs, indices, 2))
        if depths is not None:
            group_depths.append(view_along_axis(depths, indices, 2))

    for b in B_range:
        bidx = np.random.choice(blocks, size=nblocks, replace=True)
        #block_loci = np.array([index for index in block_indices[b] for b in bidx], dtype='uint32')
        #block_loci = np.where(np.in1d(block_ids, bidx))[0]
        #block_loci = np.concatenate(block_indices[bidx])
        #strap_lens.append(len(block_loci))
        #sliced_freqs = freqs[:, :, block_loci]
        #sliced_depths = depths
        sliced_freqs = np.concatenate([group_freqs[b] for b in bidx], axis=2)
        sliced_depths = None
        if depths is not None:
            #sliced_depths = depths[:, :, block_loci]
            sliced_depths = np.concatenate([group_depths[b] for b in bidx], axis=2)
        stat = estimator(freqs=sliced_freqs, depths=sliced_depths, diploids=diploids, **kwargs)
        straps.append(stat)
    straps = np.stack(straps)
    That = np.mean(straps, axis=0)
    if return_straps:
        return straps
    return bootstrap_ci(That, straps, alpha=alpha, method=ci_method)


def bootstrap_temporal_cov(freqs, depths=None, diploids=None, average_replicates=True,
                           **kwargs):
    R, T, L = freqs.shape
    covs = temporal_replicate_cov(freqs=freqs, depths=depths, diploids=diploids, **kwargs)
    temp_covs = stack_temporal_covariances(covs, R, T)
    if average_replicates:
        return temp_covs.mean(axis=2)
    return temp_covs


