## covmethods.py -- temporal/replicate covariance methods
import numpy as np

from cvtk.utils import flatten_matrix

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

def calc_hets(freqs, depths=None, diploids=None, bias=False):
    if depths is not None:
        assert(freqs.shape == depths.shape)
    if diploids is not None:
        dnrow, dncol = diploids.shape
        assert((dnrow, dncol) == (freqs.shape[0], freqs.shape[1]))
        diploids = diploids[..., np.newaxis]
    het = 2*freqs*(1-freqs)
    if not bias:
        if depths is not None:
            het *= depths / (depths-1)
        if diploids is not None:
            # additional bias correction needed
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
        if isinstance(diploids, int):
            diploids = np.repeat(diploids, R*ntimepoints)
            diploids = diploids.reshape(R, ntimepoints)
        elif isinstance(diploids, np.ndarray):
            try:
                if diploids.ndim == 1:
                    assert(diploids.shape[0] == R * ntimepoints)
                    diploids = diploids.reshape(R, ntimepoints)
                elif diploids.ndim == 2:
                    assert(diploids.shape == (R, ntimepoints))
                else:
                    assert(False)
            except AssertionError:
                msg = ("diploids must be an integer or a matrix of shape "
                       f"nreplicates x ntimepoints ({R} x {ntimepoints})")
                raise ValueError(msg)
    if depths is not None:
        assert(freqs.shape == depths.shape)
    if diploids is not None:
        assert(freqs.shape == diploids.shape)
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

def temporal_cov(freqs, depths=None, diploids=None, center=True):
    """
    Notes:
     Some sampled frequnecies can be NaN, since in numpy, 0/0 is NaN. This
     needs to be handled accordingly.
    """
    #freqs[np.isnan(freqs)] = 0
    # check the dimensions are compatable, padding on a dimension
    # for R = 1 cases
    freqs, depths, diploids = correct_dimensions(freqs, depths, diploids)
    deltas = calc_deltas(freqs)

    R, T, L = deltas.shape
    hets = calc_hets(freqs, depths=depths, diploids=diploids)
    mean_hets = hets.mean(axis=freqs.ndim-1)
    # Calculate the heterozygosity denominator, which is
    # p_min(t,s) (1-p_min(t,s)). The following function calculates
    # unbiased heterozygosity; we take ½ of it.
    het_denom = replicate_average_het_matrix(mean_hets, R, T, L) / 2.

    # With all the statistics above calculated, we can flatten everything
    # for the next calculations. This simply rolls replicates and timepoints
    # into 'samples'.
    deltas = flatten_matrix(deltas, R, T, L)
    if depths is not None:
        depths = flatten_matrix(depths, R, T+1, L)
    if diploids is not None:
        diploids = diploids.reshape((1, R*(T+1)))
    # roll out hets
    hets = hets.reshape((R * (T+1), L))

    # assert that the deltas matrix is 2D (i.e. if it's for replicate
    # temporal design, it's already been flattened to (R x T) x L matrix).
    assert(deltas.ndim == 2)
    RxT, L = deltas.shape
    if depths is not None:
        assert(depths.shape[1] == L)
    if diploids is not None:
        assert(mean_hets.shape == diploids.shape)

    # we turn the hets, diploids, and depths into 
    # multidimensional arrays TODO don't convert in first place?
    half_hets = (hets / 2.).reshape((R, T+1, L))
    if depths is not None:
    	depths = depths.reshape((R, T+1, L))
    if diploids is not None:
    	diploids = diploids.reshape((R, T+1, L))
    # calculate variance-covariance matrix
    cov = np.cov(deltas, bias=True)

    # correction arrays — these are built up depending on input
    ave_bias = np.zeros((R, (T+1)))
    var_correction = np.zeros(RxT)
    covar_correction = np.zeros(R*T-1)

    # build up correct for any combination of depth / diploid / depth & diploid
    # data
    diploid_correction = 0
    depth_correction = 0
    if depths is not None:
        depth_correction = 1 / depths
    if diploids is not None:
        diploid_correction = 1 / (2 * diploids)
        if depths is not None:
            diploid_correction += 1 / (2 * depths * diploids)
    # the bias vector for all timepoints
    ave_bias += (half_hets * (diploid_correction + depth_correction)).mean(axis=2)
    var_correction += (- ave_bias[:, :-1] - ave_bias[:, 1:]).reshape(RxT)
    # the covariance correction is a bit trickier: it's off diagonal elements, but 
    # after every Tth entry does not need a correction, as it's a between replicate
    # covariance. We append a zero column, and then remove the last element.
    covar_correction += np.hstack((ave_bias[:, 1:-1], np.zeros((R, 1)))).reshape(R*T)[:-1]

    cov += (np.diag(var_correction) +
            np.diag(covar_correction, k=1) +
            np.diag(covar_correction, k=-1))

    return cov / het_denom

