## covmethods.py -- temporal/replicate covariance methods
import numpy as np

from cvtk.utils import flatten_matrix

def calc_deltas(freqs):
    """
    Calculates the deltas matrix, which is the frequency matrix where
    entry out[n] = x[n+1] - x[n]. This takes a R x T x L frequency array.
    """
    assert(freqs.ndim == 3)
    return np.diff(freqs, axis=1)

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

def temporal_cov(freqs, depths=None, diploids=None, center=False,
                 N=None, weighted=False):
    """
    Notes:
     Some sampled frequnecies can be NaN, since in numpy, 0/0 is NaN. This
     needs to be handled accordingly.
    """
    freqs[np.isnan(freqs)] = 0
    # check the dimensions are compatable
    freqs, depths, diploids = correct_dimensions(freqs, depths, diploids)
    hets = calc_hets(freqs, depths, diploids=diploids)
    #mean_hets = calc_mean_hets(freqs, depths, diploids=diploids)
    mean_hets = np.nanmean(hets, axis=freqs.ndim-1)
    deltas = calc_deltas(freqs)

    R, T, L = deltas.shape
    deltas = flatten_matrix(deltas, R, T, L)
    if depths is not None:
        depths = flatten_matrix(depths, R, T+1, L)
    if diploids is not None:
        diploids = diploids.reshape((1, R*(T+1)))

    # Calculate the heterozygosity denominator, which is
    # p_min(t,s) (1-p_min(t,s)). The following function calculates
    # unbiased heterozygosity; we take ½ of it.
    het_denom = replicate_average_het_matrix(mean_hets, R, T, L) / 2.

    # assert that the deltas matrix is 2D (i.e. if it's for replicate
    # temporal design, it's already been flattened to (R x T) x L matrix).
    assert(deltas.ndim == 2)
    RxT, L = deltas.shape
    if depths is not None:
        assert(depths.shape[1] == L)
    if diploids is not None:
        assert(mean_hets.shape == diploids.shape)

    half_hets = mean_hets / 2.
    # calculate variance-covariance matrix
    if center:
        if weighted:
            raise ValueError("if center=True, weighted must equal False")
        cov = np.cov(deltas, bias=True)
    else:
        # note: no bias correction here because we don't center
        if weighted:
            if depths is None:
                raise ValueError("if weighted=True, depths must specified")
            w = (depths[1:, :] + depths[:-1, :]) / 2
            w /=  np.broadcast_to(np.sum(w, axis=1)[:, np.newaxis], deltas.shape)
            weighted_deltas = np.sqrt(w) * deltas   # TODO DOUBLE CHECK
            #pdb.set_trace()
            cov = (weighted_deltas @ weighted_deltas.T)
        else:
            cov = (deltas @ deltas.T) / L

    # correction arrays — these are built up depending on input
    var_correction = np.repeat(0., RxT)
    covar_correction = np.repeat(0., RxT-1)

    # the follow are p0(1-p0) and p1(1-p1) for each Δp
    # NOTE: the ... indexing ensures this works for any R x T x L array
    # or T x L matrix. Dimensionality of half_hets is R x T since it's been
    # averaged over loci.
    half_het0 = half_hets[..., :-1].flatten()
    half_het1 = half_hets[..., 1:].flatten()
    if depths is not None:
        depths = np.nanmean(depths, axis=1)  # average over loci
        depth0 = depths[..., :-1]
        depth1 = depths[..., 1:]
        depth_correction = - half_het0 / depth0 - half_het1 / depth1
        #sample_correction = - np.nanmean(half_hets[:-1, :] / depth[:-1, :], axis=1) - np.nanmean(half_hets[1:, :] / depth[1:, :], axis=1)
        var_correction += depth_correction
        # note we use half_het1 and depth1 here: the shared error term of
        # cov(Δp_{t}, Δp_{t+1}) is the t+1 term
        covar_correction += half_het1[:-1] / depth1[:-1]
    if diploids is not None:
        # TODO: comeback to whether flatten is the best thing to do here
        # NOTE: factor of two is because two chromosomes are sampled
        # from the population for each diploid.
        diploids0 = 2 * diploids[..., :-1].flatten()
        diploids1 = 2 * diploids[..., 1:].flatten()
        diploid_correction = - half_het0/diploids0 - half_het1/diploids1
        # TODO DOUBLE CHECK, specifically the time index
        covar_diploid_correction = half_het1[:-1]/diploids1[:-1]
        if depths is not None:
            diploid_correction += - half_het0/(diploids0 * depth0) - half_het1/(diploids1 * depth1)
            covar_diploid_correction += half_het1[:-1]/(diploids1[:-1] * depth1[:-1])
        var_correction += diploid_correction
        covar_correction += covar_diploid_correction
    if N is not None:
        var_correction += half_het0 / N
    #pdb.set_trace()
    # create correction matrix
    diag_correction = np.diag(var_correction)
    offdiag_correction = np.diag(covar_correction, k=1) + np.diag(covar_correction, k=-1)
    return cov / het_denom + diag_correction / het_denom + offdiag_correction / het_denom



def corrected_cov(deltas, mean_hets, depths=None, center=False, N=None,
                  weighted=False, diploids=None):
    """
    Calculate a corrected replicate-temporal covariance matrix.

    All corrections *only* apply to the diagonal elements and off-diagonal
    elements, by the way the block matrices are set up. The block matrices
    along the diagonal are all temporal covariance matrices.

    TODO:
      - shared noise on replicate covariance matrices?
    """
    # assert that the deltas matrix is 2D (i.e. if it's for replicate
    # temporal design, it's already been flattened to (R x T) x L matrix).
    assert(deltas.ndim == 2)
    RxT, L = deltas.shape
    if depths is not None:
        assert(depths.shape[1] == L)
    if diploids is not None:
        assert(mean_hets.shape == diploids.shape)

    half_hets = mean_hets / 2.
    # calculate variance-covariance matrix
    if center:
        if weighted:
            raise ValueError("if center=True, weighted must equal False")
        cov = np.cov(deltas, bias=True)
    else:
        # note: no bias correction here because we don't center
        if weighted:
            if depths is None:
                raise ValueError("if weighted=True, depths must specified")
            w = (depths[1:, :] + depths[:-1, :]) / 2
            w /=  np.broadcast_to(np.sum(w, axis=1)[:, np.newaxis], deltas.shape)
            weighted_deltas = np.sqrt(w) * deltas   # TODO DOUBLE CHECK
            #pdb.set_trace()
            cov = (weighted_deltas @ weighted_deltas.T)
        else:
            cov = (deltas @ deltas.T) / L

    # correction arrays — these are built up depending on input
    var_correction = np.repeat(0., RxT)
    covar_correction = np.repeat(0., RxT-1)

    # the follow are p0(1-p0) and p1(1-p1) for each Δp
    # NOTE: the ... indexing ensures this works for any R x T x L array
    # or T x L matrix. Dimensionality of half_hets is R x T since it's been
    # averaged over loci.
    half_het0 = half_hets[..., :-1].flatten()
    half_het1 = half_hets[..., 1:].flatten()
    if depths is not None:
        depths = np.nanmean(depths, axis=1)  # average over loci
        depth0 = depths[..., :-1]
        depth1 = depths[..., 1:]
        depth_correction = - half_het0 / depth0 - half_het1 / depth1
        #sample_correction = - np.nanmean(half_hets[:-1, :] / depth[:-1, :], axis=1) - np.nanmean(half_hets[1:, :] / depth[1:, :], axis=1)
        var_correction += depth_correction
        # note we use half_het1 and depth1 here: the shared error term of
        # cov(Δp_{t}, Δp_{t+1}) is the t+1 term
        covar_correction += half_het1[:-1] / depth1[:-1]
    if diploids is not None:
        # TODO: comeback to whether flatten is the best thing to do here
        diploids0 = 2 * diploids[..., :-1].flatten()
        diploids1 = 2 * diploids[..., 1:].flatten()
        diploid_correction = - half_het0/diploids0 - half_het1/diploids1
        # TODO DOUBLE CHECK, specifically the time index
        covar_diploid_correction = half_het1[:-1]/diploids1[:-1]
        if depths is not None:
            diploid_correction += - half_het0/(diploids0 * depth0) - half_het1/(diploids1 * depth1)
            covar_diploid_correction += half_het1[:-1]/(diploids1[:-1] * depth1[:-1])
        var_correction += diploid_correction
        covar_correction += covar_diploid_correction
    if N is not None:
        var_correction += half_het0 / N
    #pdb.set_trace()
    # create correction matrix
    diag_correction = np.diag(var_correction)
    offdiag_correction = np.diag(covar_correction, k=1) + np.diag(covar_correction, k=-1)
    return cov / half_het0 + diag_correction / half_het0 + offdiag_correction / half_het0



def replicate_temporal_cov(freqs, depths=None, diploids=None, *args, **kwargs):
    R, T, L = freqs.shape

    freqs[np.isnan(freqs)] = 0
    freqs, depths, diploids = correct_dimensions(freqs, depths, diploids)
    ntimepoints, L = freqs.shape
    hets = calc_hets(freqs, depths, diploids=diploids)
    deltas = calc_deltas(freqs)

    # validate hets dimensions
    assert(hets.shape[1] == T+1)
    het_denom = hets[:,:-1] / 2
    assert(het_denom.shape[0] == R)
    assert(het_denom.shape[1] == T)


    # we want to calculate covariances across all samples ∈ (timepoints x
    # replicates), so we first flatten the first dimension. See note on this
    # above TODO (in old code)
    M = deltas.reshape((R * T, L))

    return corrected_cov(deltas, hets, depths=depths, diploids=diploids, **kwargs)


