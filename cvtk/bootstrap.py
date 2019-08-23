from itertools import chain
import numpy as np
from tqdm import tnrange
from cvtk.utils import view_along_axis
from cvtk.cov import stack_temporal_covariances, temporal_replicate_cov


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
    """
    Wrapper around temporal_replicate_cov() that takes temporal-replicate covariance matrix
    and reshapes it to an T x T x R temporal covariance array. If average_replicates=True,
    this averages across all replicates.
    """
    R, Tp1, L = freqs.shape  # freqs is R x (T+1) x L
    covs = temporal_replicate_cov(freqs=freqs, depths=depths, diploids=diploids, 
                                  **kwargs)
    temp_covs = stack_temporal_covariances(covs, R, Tp1 - 1)
    if average_replicates:
        return temp_covs.mean(axis=2)
    return temp_covs


def block_bootstrap_ratio_averages(blocks_numerator, blocks_denominator, 
                                   block_indices, block_seqids, B, estimator, 
                                   depths=None, diploids=None, 
                                   alpha=0.05, keep_seqids=None, return_straps=False,
                                   ci_method='pivot', progress_bar=False, **kwargs):
    """
    This block bootstrap is used often for quantities we need to calculate that are 
    ratios of expectations, e.g. standardized temporal covariance (cov(Δp_t, Δp_s) / p_t(1-p_t))
    and G, both of which are expectations over loci. We use the linearity of expectation
    to greatly speed up the block bootstrap procedure.

    We do so by pre-calculating the expected numerator and denominator for each block, 
    and then take a weighted average over the bootstrap sample for each the numerator and 
    denominator, and then take the final ratio.

    It's assumed that blocks_numerator and blocks_denominator are both multidimension arrays
    with the first dimension being the block (e.g. tile) dimension.
    """
    if progress_bar:
        B_range = tnrange(int(B), desc="bootstraps")
    else:
        B_range = range(int(B))

    # We create the vector of indices to sample with replacement from, excluding 
    # any blocks with seqids not in keep_seqids.
    if keep_seqids is not None:
        blocks = np.array([i for i, seqid in enumerate(block_seqids) 
                           if seqid in keep_seqids], dtype='uint32')
    else:
        blocks = np.array([i for i, seqid in enumerate(block_seqids)], dtype='uint32')

    # Calculate the weights 
    weights = np.array([len(x) for x in block_indices]) 
    weights = weights/weights.sum()

    # number of samples in resample
    nblocks = len(blocks)
    straps = list()
    group_freqs = list()
    group_depths = None if depths is None else []
    
    for b in B_range:
        bidx = np.random.choice(blocks, size=nblocks, replace=True)
        exp_numerator = weighted_mean(blocks_numerator[bidx, ...], weights=weights[bidx])
        exp_denominator = weighted_mean(blocks_denominator[bidx, ...], weights=weights[bidx])
        stat = estimator(exp_numerator, exp_denominator, **kwargs)
        straps.append(stat)
    straps = np.stack(straps)
    That = np.mean(straps, axis=0)
    if return_straps:
        return straps
    return bootstrap_ci(That, straps, alpha=alpha, method=ci_method)


def cov_estimator(cov, het_denom, R, T, average_replicates=False, warn=False):
    warn_type = 'ignore' if not warn else 'warn'
    with np.errstate(divide=warn_type, invalid=warn_type):
        cov = cov / het_denom
    if not average_replicates:
        return cov
    else:
        return np.mean(stack_temporal_covariances(cov, R, T), axis=2)

