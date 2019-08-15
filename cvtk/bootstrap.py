import numpy as np

def bootstrap_ci(estimate, straps, alpha=0.05, method='pivot'):
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
        return qlower, estimate, qupper
    elif method == 'pivot':
        return 2*estimate - qupper, estimate, 2*estimate - qlower
    else:
        raise ValueError("method must be either 'pivot' or 'percentile'")


def block_bootstrap_temporal_covs(covs, block_indices, block_seqids, B, alpha=0.05, 
                                  bootstrap_replicates=False,
                                  replicate=None, average_replicates=False, 
                                  keep_seqids=None, return_straps=False, ci_method='pivot'):
    """
    Bootstrap the temporal covariances. This procedure bootstraps the temporal sub-block 
    covariance matrices (there are R of these, and each is TxT).

    Note that the bootstrap of the covariances takes pre-existing covariances matrices per 
    block (e.g. tile), and does a *weighted* average of these covariances, with weights
    determined by the number of loci in the block.

    Params: 
           - covs: temporal covariance 4D array, nblock x T x T x R.
           - block_indices: list of lists, each inner list contains the indices for 
               the SNPs for that block.
           - block_seqids: list of seqids for each block.
           - B: number of bootstraps
           - alpha: α level
           - bootstrap_replicates: whether the R replicates are resampled as well, and 
              covariance is averaged over these replicates.
           - replicate: only bootstrap the covariances for a single replicate (cannot be used 
              with bootstrap_replicates).
           - average_replicates: whether to average across all replicates.
           - keep_seqids: which seqids to include in bootstrap
           - return_straps: whether to return the actual bootstrap vectors.
           - ci_method: 'pivot' or 'percentile'
    
    Future improvements:
           - use any arbitrary masked array function for the statistic, not just the mean.
    """
    assert(covs.ndim == 4)
    nblocks, T, T_, R = covs.shape
    assert(T == T_)
    if replicate is not None and bootstrap_replicates:
            msg = "cannot bootstrap on single replicate; set either bootstrap_replicates=False or replicate=None"
            raise ValueError(msg)
    if replicate is not None:
        covs = covs[:, :, :, replicate]
    # block weights by number of loci
    assert(not isinstance(keep_seqids, str))  # prevent a common error
    indices_seqid_pairs = zip(block_indices, block_seqids)
    weights = np.array([len(x) for x, seqid in indices_seqid_pairs]) 
    weights = weights/weights.sum()
    covs_idx = np.arange(nblocks)
    if keep_seqids is not None:
        keep_seqids = set(keep_seqids)
        covs_idx = np.array([i for i, seqid in enumerate(indices_seqid_pairs) if seqid in keep_seqids])
    # prune down the index if not all seqids kept
    weights = weights[covs_idx]
    covs = covs[covs_idx, ...]
    nblocks = covs.shape[0]
    
    # number of samples in resample
    straps = list()
    for b in np.arange(B):
        bidx = np.random.randint(0, nblocks, size=nblocks)
        # get the windows of the resampled indices
        mat = covs[bidx, ...]
        if bootstrap_replicates:
            #assert(replicate is None)
            ridx = np.random.randint(0, R, size=R)
            mat = mat[:, :, :, ridx]
        # mask the covariance matrix, since ma.average is the only nanmean
        # in numpy that takes weights
        covs_masked = np.ma.masked_array(mat, np.isnan(mat))
        avecovs = np.ma.average(covs_masked, axis=0, weights=weights).data
        if average_replicates:
            avecovs = avecovs.mean(axis=2)
        straps.append(avecovs)
    straps = np.stack(straps)
    That = np.mean(straps, axis=0)
    if return_straps:
        return straps
    return bootstrap_ci(That, straps, alpha=alpha, method=ci_method)
