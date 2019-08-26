import warnings
import numpy as np
from tqdm import tqdm, tqdm_notebook

from cvtk.utils import flatten_matrix, view_along_axis

def calc_deltas(freqs):
    """
    Calculates the deltas matrix, which is the frequency matrix where
    entry out[n] = x[n+1] - x[n]. This takes a R x T x L frequency array.
    """
    if freqs.ndim == 2:
        return np.diff(freqs, axis=0)
    elif freqs.ndim == 3:
        return np.diff(freqs, axis=1)
    else:
        raise ValueError("eqs.ndim must be eitehr 2 or 3")

def calc_hets(freqs, depths=None, diploids=None, bias=False, warn=False):
    R, T, L = freqs.shape
    if depths is not None:
        assert(freqs.shape == depths.shape)
    het = 2*freqs*(1-freqs)
    warn_type = 'ignore' if not warn else 'warn'
    with np.errstate(divide=warn_type, invalid=warn_type):
        if not bias:
            if depths is not None:
                het *= depths / (depths-1)
            if diploids is not None:
                het *= 2*diploids / (2*diploids - 1)
    return het


def calc_mean_hets(freqs, depths=None, diploids=None, bias=False):
    hets = calc_hets(freqs, depths=depths, diploids=diploids, bias=bias)
    return np.nanmean(hets, axis=freqs.ndim-1)


def correct_dimensions(freqs, depths=None, diploids=None):
    # pad on an extra dimensions for 1-replicate designs
    if freqs.ndim == 2:
        freqs = freqs[np.newaxis, ...]
    if depths is not None:
        if depths.ndim == 2:
            depths = depths[np.newaxis, ...]
        assert(freqs.shape == depths.shape)

    R, ntimepoints, L = freqs.shape
    if diploids is not None:
        depth_is_single_int = diploids.size == 1
        depth_is_valid_vector = diploids.ndim == 2 and diploids.shape == (R, ntimepoints)
        depth_is_valid_array = diploids.ndim == 3 and diploids.shape == (R, ntimepoints, 1)
        if not (depth_is_single_int or depth_is_valid_vector or depth_is_valid_array):
            msg = ("diploids must be an integer or a matrix of shape "
                   f"nreplicates x ntimepoints ({R} x {ntimepoints})")
            raise ValueError(msg)
    return freqs, depths, diploids


def temporal_block_matrix_indices(R, T):
    """
    Return the indices of times t, s in each block matrix, where there are
    R x R block matrices (one for each replicate), and each block matrix is
    a covariance matrix of rows t, columns s timediffs.
    """
    row, col = replicate_block_matrix_indices(T, 1)
    rows = np.tile(row, (R, R))
    cols = np.tile(col, (R, R))
    return rows, cols

def replicate_block_matrix_indices(R, T):
    """
    Build block matrices of row and colum indices indicating the *replicate*.

    Each block matrix is (R x T) x (R x T), with each block being T x T.
    The structure is

    [[ A/A  A/A  A/C ]
     [ B/A  B/B  B/C ]
     [ C/A  C/B  C/C ]]

    with X/Y indicating replicate X, replicate Y blocks. The two matrices
    returned are rows (all the X values) and cols (all y values).
    """
    row_bm = np.vstack([np.full((T, R*T), i) for i in range(R)])
    col_bm = row_bm.T
    return (row_bm, col_bm)

def stack_replicate_covariances(covmat, R, T, stack=True, return_tuple=False,
                                upper_only=True):
    """
    Upper only now.
    """
    layers = []
    rows, cols = replicate_block_matrix_indices(R, T)
    for i in np.arange(R):
        for j in np.arange(R):
            if i == j:
                # ignore temporal covs
                continue
            if upper_only and i < j:
                continue
            this_block_matrix = np.logical_and(rows == i, cols == j)
            block = covmat[this_block_matrix].reshape(T, T)
            if stack:
                layers.append(block)
            else:
                if return_tuple:
                    # store i, j
                    layers.append((i, j, block))
                else:
                    layers.append(block)
    if stack:
        return np.stack(layers).T
    return layers

def stack_temporal_covariances(covmat, R, T, stack=True):
    """
    Stack temporal sub-matrices of the temporal covariance matrix.
    """
    layers = []
    rows, cols = replicate_block_matrix_indices(R, T)
    for i in np.arange(R):
        this_block_matrix = np.logical_and(rows == i, cols == i)
        block = covmat[this_block_matrix].reshape(T, T)
        layers.append(block)
    if stack:
        return np.stack(layers).T
    return layers


def replicate_average_het_matrix(hets, R, T, L):
    """
    Create the heterozygosity denominator for the temporal-replicate
    variance covariance matrix.

    Create the heterozygosity denominator, which is of block form.
    Each block is a replicate. Each element is,

    (p_{A, min(t,s)}(1-p_{A, min(t,s)}) +
         p_{B, min(t,s)}(1-p_{B, min(t,s)})) / 2

    For temporal block covariance matrices (those along the diagonal), this is
    the same as the usual p_min(t,s) (1 - p_min(t,s)). For replicate blocks,
    we average the two initial frequencies.
    """
    assert(hets.shape == (R, T+1))
    # Create the heterozygosity denominator, which is of block form.
    # Each block is a replicate. Each element is,
    # (p_{A, min(t,s)}(1-p_{A, min(t,s)}) +
    #      p_{B, min(t,s)}(1-p_{B, min(t,s)})) / 2
    # for each replicate. We use some numpy tricks here.
    time_indices = np.arange(0, T)
    # this is the T x T matrix of min(s,t):
    min_t_s = np.minimum.outer(time_indices, time_indices)
    # next, we make the min(t,s) matrix of indices for each of the
    # replicate submatrices of the block matrix. There are R x R
    # block matrices, each block submatrix is T x T. This makes R x R
    # min(t,s) matrices to be used for indices:
    min_t_s_block = np.tile(min_t_s, (R, R))
    # now, we build indices for row/col for block matrices
    # which are used to index the heterozygosity denominators for
    # the replicated that are two be averaged:
    row_bm, col_bm = replicate_block_matrix_indices(R, T)
    # use indexing vectors to get heterozygosities
    A = hets[row_bm.ravel(), min_t_s_block.ravel()]
    B = hets[col_bm.ravel(), min_t_s_block.ravel()]
    avehet_min = (A + B).reshape((R*T, R*T)) / 2
    return avehet_min


def var_by_group(groups, freqs, t=None, depths=None, diploids=None, 
                 bias_correction=True, deltas=None, progress_bar=False,
                 standardize=True):
    group_depths, group_diploids, group_deltas = None, None, None
    vars = []
    groups_iter = groups
    if progress_bar:
        groups_iter = tqdm_notebook(groups)
    for indices in groups_iter:
        group_freqs = view_along_axis(freqs, indices, 2)
        if depths is not None:
            group_depths = view_along_axis(depths, indices, 2)
        #if diploids is not None:
        #    group_diploids = view_along_axis(diploids, indices, 2)
        if deltas is not None:
            group_deltas = view_along_axis(deltas, indices, 2)
        tile_vars = total_variance(group_freqs,
                                   depths=group_depths, 
                                   diploids=group_diploids,
                                   t=t, standardize=standardize,
                                   bias_correction=bias_correction)
        vars.append(tile_vars)
    return vars


def cov_by_group(groups, freqs, depths=None, diploids=None, standardize=True,
                 bias_correction=True, deltas=None, use_masked=False, 
                 share_first=False, return_ratio_parts=False,
                 progress_bar=False):
    group_depths, group_deltas = None, None
    covs = []
    het_denoms = [] # incase return_ratio_parts=True
    groups_iter = groups
    if progress_bar:
        groups_iter = tqdm_notebook(groups)
    for indices in groups_iter:
        group_freqs = view_along_axis(freqs, indices, 2)
        if depths is not None:
            group_depths = view_along_axis(depths, indices, 2)
        #if diploids is not None:
        #    group_diploids = view_along_axis(diploids, indices, 2)
        if deltas is not None:
            group_deltas = view_along_axis(deltas, indices, 2)
        res = temporal_replicate_cov(group_freqs,
                                     depths=group_depths, 
                                     diploids=diploids,
                                     bias_correction=bias_correction, 
                                     standardize=standardize,
                                     use_masked=use_masked,
                                     share_first=share_first,
                                     return_ratio_parts=return_ratio_parts,
                                     deltas=group_deltas)
        if return_ratio_parts:
            cov, het_denom = res
            covs.append(cov)
            het_denoms.append(het_denom)
        else:
            covs.append(res)
    if return_ratio_parts:
        return covs, het_denoms
    return covs


def stack_temporal_covs_by_group(covs, R, T, stack=True, **kwargs):
    res = [stack_temporal_covariances(c, R, T, stack=stack, **kwargs) for c in covs]
    if stack:
        return np.stack(res)
    return res

def stack_replicate_covs_by_group(covs, R, T, stack=True, **kwargs):
    res = [stack_replicate_covariances(c, R, T, stack=stack, **kwargs) for c in covs]
    if stack:
        return np.stack(res)
    return res



def stack_replicate_covs_by_group(covs, R, T, stack=True, **kwargs):
    res = [stack_replicate_covariances(c, R, T, stack=stack, **kwargs) for c in covs]
    if stack:
        return np.stack(res)
    return res



def temporal_replicate_cov(freqs, depths=None, diploids=None, center=True, 
                           bias_correction=True, standardize=True, deltas=None, 
                           use_masked=False, share_first=False, 
                           return_ratio_parts=False, warn=False):
    """
    Params:
      ...
      - deltas: optional deltas matrix (e.g. if permuted deltas are used). If
          this is None, it's calculated in this function.
    Notes:
     Some sampled frequnecies can be NaN, since in numpy, 0/0 is NaN. This
     needs to be handled accordingly.
    """
    warn_type = 'ignore' if not warn else 'warn'
    #freqs[np.isnan(freqs)] = 0
    # check the dimensions are compatable, padding on a dimension
    # for R = 1 cases
    freqs, depths, diploids = correct_dimensions(freqs, depths, diploids)
    if deltas is None:
        deltas = calc_deltas(freqs)

    R, T, L = deltas.shape
    hets = calc_hets(freqs, depths=depths, diploids=diploids)
    # Calculate the heterozygosity denominator, which is
    # p_min(t,s) (1-p_min(t,s)). The following function calculates
    # unbiased heterozygosity; we take ½ of it.
    mean_hets = np.nanmean(hets, axis=freqs.ndim-1)
    het_denom = replicate_average_het_matrix(mean_hets, R, T, L) / 2.

    # With all the statistics above calculated, we can flatten the deltas
    # matrix # for the next calculations. This simply rolls replicates and 
    # timepoints into 'samples'.
    deltas = flatten_matrix(deltas, R, T, L)

    # Assert that the deltas matrix is 2D (i.e. if it's for replicate
    # temporal design, it's already been flattened to (R x T) x L matrix)
    # and is of right dimension. 
    assert(deltas.ndim == 2)
    RxT, L = deltas.shape
    # The depths and diploids matrices are kept
    # in 3D, as thse corrections only affect the temporal covariance 
    # submatrices along the diagonal.
    if depths is not None:
        assert(depths.shape == (R, T+1, L))
    if diploids is not None:
        assert(diploids.shape == (R, T+1, 1))

    # calculate variance-covariance matrix
    if use_masked:
        deltas_masked = np.ma.masked_invalid(deltas)
        cov = np.ma.cov(deltas_masked, bias=True).data
    else:
        cov = np.cov(deltas, bias=True)

    if not bias_correction:
        if return_ratio_parts:
            return cov, het_denom
        if standardize:
            with np.errstate(divide=warn_type, invalid=warn_type):
                cov = cov / het_denom
        return cov
        

    # correction arrays — these are built up depending on input
    ave_bias = np.zeros((R, (T+1)))
    var_correction = np.zeros(RxT)
    covar_correction = np.zeros(R*T-1)

    # build up correct for any combination of depth / diploid / depth & diploid
    # data
    diploid_correction = 0.
    depth_correction = 0.

    with np.errstate(divide=warn_type, invalid=warn_type):
        if depths is not None:
            depth_correction = 1 / depths
        if diploids is not None:
            diploid_correction = 1 / (2 * diploids)
            if depths is not None:
                diploid_correction = diploid_correction + 1 / (2 * depths * diploids)
    # the bias vector for all timepoints
    ave_bias += np.nanmean(0.5 * hets * (diploid_correction + depth_correction), axis=2)
    var_correction += (- ave_bias[:, :-1] - ave_bias[:, 1:]).reshape(RxT)
    if share_first:
        #import pdb; pdb.set_trace()
        cov = cov - ave_bias[0, 0]
    # the covariance correction is a bit trickier: it's off diagonal elements, but 
    # after every Tth entry does not need a correction, as it's a between replicate
    # covariance. We append a zero column, and then remove the last element.
    covar_correction += np.hstack((ave_bias[:, 1:-1], np.zeros((R, 1)))).reshape(R*T)[:-1]

    cov += (np.diag(var_correction) +
            np.diag(covar_correction, k=1) +
            np.diag(covar_correction, k=-1))

    if standardize:
        if return_ratio_parts:
            return cov, het_denom
        with np.errstate(divide=warn_type, invalid=warn_type):
            cov = cov / het_denom
    return cov


def total_variance(freqs, depths=None, diploids=None, t=None, standardize=True, 
                   bias_correction=True, warn=False):
    """
    Calculate the Var(p_t - p_0) across all replicates.
    """
    R, ntimepoints, L = freqs.shape
    if t is None:
        t = ntimepoints-1
    assert(t < ntimepoints)
    pt_p0 = (freqs[:, t, :] - freqs[:, 0, :])
    var_pt_p0 = pt_p0.var(axis=1)

    if not bias_correction:
        if standardize:
            return var_pt_p0 / np.nanmean(hets[:, 0, :], axis=1)
        return var_pt_p0 

    diploid_correction = 0.
    depth_correction = 0.
    var_correction = 0.
    ave_bias = 0.
    warn_type = 'ignore' if not warn else 'warn'
    with np.errstate(divide=warn_type, invalid=warn_type):
        if depths is not None:
            depth_correction = 1 / depths[:, (0, t), :]
        if diploids is not None:
            diploid_correction = 1 / (2 * diploids[:, (0, t), :])
            if depths is not None:
                b = 1 / (2 * depths[:, (0, t), :] * diploids[:, (0, t), :])
                diploid_correction = diploid_correction + b
    # the bias vector for all timepoints
    hets = calc_hets(freqs, depths=depths, diploids=diploids)[:, (0, t), :]
    ave_bias += np.nanmean(0.5 * hets * (diploid_correction + depth_correction), axis=2)
    var_correction += (- ave_bias[:, 0] - ave_bias[:, 1])
    out =  var_pt_p0 + var_correction
    # in some cases, subtracting off the expected bias leads us to create negative
    # covariances. In th and ese cases, we don't apply the correction.
    if warn and np.any(out <= 0):
        msg = "Some bias-corrected variances were negative!"
        warnings.warn(msg)
    if standardize:
        out = out / np.nanmean(hets[:, 0, :], axis=1)
    return out
 
