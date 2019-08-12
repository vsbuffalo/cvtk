## covmeths.py -- temporal covariance classes and data processing methods
"""

Notation:

S: number of samples, S = R x T
R: number of replicates
ntimepoints: number of temporal samples
T: number of Δp, which is ntimepoints - 1
L: number of loci

We assume the incoming frequency matrix is S x L, where the design is balanced
so R is constant across the number of timepoints. The function
reshape_freqs_to_tensor() reshapes the S x L matrix to a R x T x L tensor.

"""
from collections import defaultdict, OrderedDict, Counter
from operator import itemgetter
from itertools import groupby
import pickle
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tqdm import tnrange, tqdm_notebook
from cvtk.covmethods import replicate_block_matrix_indices
from cvtk.covmethods import calc_temporal_replicate_covs, calc_mean_hets
from cvtk.covmethods import cov_tensor_to_dataframe, stack_temporal_covariances
from cvtk.covmethods import stack_replicate_covariances
from cvtk.covmethods import cov_matrices_to_dataframe, calc_G, calc_rep_G
from cvtk.covmethods import permute_delta_matrix, reshape_empirical_null
from cvtk.covmethods import covs_by_group
from cvtk.misc import view_along_axis, unnest, squeeze_arrays, sliceify


#def slice_indices(tile_indices, tile_seqids, keep_seqids=None, by_seqid=False):
#    if by_seqid is not None and keep_seqids is not None:
#        raise ValueError("cannot use both keep_seqids and by_seqid")
#    if keep_seqids is not None:
#        keep_seqids = set(keep_seqids)
#        indices = [tile for tile, seqid in list(zip(tile_indices, tile_seqids))
#                    if seqid in keep_seqids]
#    else:
#        indices = tile_indices
#    if by_seqid:
#        indices = [[x[0] for x in v] for k, v in
#                    groupby(enumerate(self.gintervals.seqid), lambda x: x[1])]
#    return indices


def slice_loci(seqids, loci_indices=None, exclude_seqids=None):
    """
    seqids is a vector of the Sequence ID for each locus, in order of the loci
    of the array to be sliced.
    """
    loci_slice = slice(None)
    if exclude_seqids is None:
        exclude_seqids = set()
    if loci_indices is None:
        idx = [i for i, seqid in enumerate(seqids) if seqid not in exclude_seqids]
    else:
        loci_indices = set(loci_indices)
        idx = [i for i, seqid in enumerate(seqids) if seqid not in exclude_seqids and i in loci_indices]
    #loci_slice = np.array(idx)
    return sliceify(idx)

def is_polymorphic(x):
    "Check if a frequency is polymorphic. If nan, is False."
    return np.logical_and(x > 0., x < 1.)

def is_heterozygous(x):
    "Check if heterozygosity is valid. If nan, is False."
    return np.logical_and(x > 0., x <= 0.5)

def sort_samples(samples):
    """
    Sort the samples, ordering and grouping by replicate, then timepoint.
    This returns a tuple, of the ordered samples and then an indexing array
    to rearrange the columns of the input frequency matrix.
    """
    sorted_i = sorted(range(len(samples)),
                      key=lambda i: (samples[i][0], samples[i][1]))
    return [samples[i] for i in sorted_i], sorted_i

def process_samples(freqs, samples):
    """
    Validate samples, ensuring they are:
    (1) the right type, a list of tuples, each tuple being
        (replicate, timepoint).
    (2) that samples are grouped by replicate and then timepoint.
    (3) the number of samples is equal to the number of rows of the
        frequency matrix.
    (4) the design is balanced (R is same for all timepoints)

    Then, if everything is proper, return a tuple of lists of each replicates
    and timepoints.

    """
    samples = list(samples)
    # check types
    try:
        assert(isinstance(samples, list))
        assert(all(isinstance(x, tuple) and len(x) == 2) for x in samples)
    except AssertionError:
        raise ValueError("samples must be a list of tuples, (replicate, timepoint).")

    # check sorting
    try:
        assert(sorted(samples, key=lambda x: (x[0], x[1])) == samples)
    except AssertionError:
        raise ValueError("samples must sorted by replicate, then timepoint, e.g. "
                        "(R1, T1), (R1, T2), (R1, T3), (R2, T1), (R2, T2), ...")

    # check dimension compatability
    if len(samples) != freqs.shape[0]:
        raise ValueError("len(samples) != number of rows in samples.")
    replicates, timepoints = zip(*samples)

    # check if design (ntimepoints/nreplicates is balanced)
    timepoints_counts = Counter(timepoints)
    replicates_counts = Counter(replicates)
    timepoints_is_balanced = len(set(timepoints_counts.values())) == 1
    replicates_is_balanced = len(set(replicates_counts.values())) == 1
    try:
        assert(timepoints_is_balanced)
    except AssertionError:
        msg = "timepoints are not balanced — equal number of replicates are needed"
        raise ValueError(msg)
    try:
        assert(timepoints_is_balanced)
    except AssertionError:
        msg = "replicates are not balanced — equal number of timepoints needed"
        raise ValueError(msg)
    return (np.array(replicates), np.array(timepoints),
            len(replicates_counts), len(timepoints_counts))


def reshape_freqs_to_tensor(freqs, R):
    """

    Take a S x L frequency matrix (where S = R x T is the number of samples),
    and the number of replicates, and reshape the frequency matrix such that
    it's a R x T x L tensor.

    """
    # find the indices where the replicate switches -- as evident from checks
    # this relies on replicates and freqs in same order, sorted by replicates!
    return np.stack(np.vsplit(freqs, R))


def filter_freqs(mat, N=None, min_af=0.0, depth_limits=None, verbose=True):
    # we ensure that we are not using a view, which would change the original
    # matrix
    freqs = mat.copy()
    tot = np.prod(freqs.shape)
    if depth_limits is not None:
        assert(N is not None and freqs.shape == N.shape)
        min_N, max_N = depth_limits
        assert(0 < min_N < max_N)
        # get average depth at each locus, averaging across samples
        Nr = N.mean(axis=(0, 1))
        N_filter = (np.logical_or(Nr < min_N, Nr > max_N))
        freqs[:, :, N_filter] = np.nan
    if min_af is not None:
        assert(0 <= min_af < 0.5)
        remove = np.logical_or(freqs <= min_af, freqs >= 1-min_af)
        if verbose:
            nr = remove.sum()
            print(f"removing {nr} ({np.round(nr/tot, 2)*100}%) loci <= {min_af} or >= {1-min_af}")
        freqs[remove] = np.nan
    return freqs

def swap_alleles(freqs, flips=None, force=False):
    """
    Swap alleles of a frequency tensor that's R x T x L. Returns a tuple of
    (freq tensor with alleles randomly swapped, L length array of which loci
    were swapped)

    If flips is not None, this is an L length array indicating which loci to
    swap (useful for debugging).
    """
    R, T, L = freqs.shape
    if flips is None:
        flips = np.random.binomial(1, 0.5, (1, L))
    try:
        assert(flips.size == L)
    except AssertionError:
        msg = f"number of flips must be equal to number of loci ({L})"
        raise ValueError(msg)
    swapped_alleles = flips[0]
    # this uses broadcasting; flips is 1 x L
    # freqs is R x T x L
    # flips is left padded 1 x 1 x L and then expanded
    # to R x T x L and subtracted off
    swapped_freqs = np.abs(flips - freqs)
    return swapped_freqs, swapped_alleles

def calc_deltas(freqs):
    """
    Calculates the deltas matrix, which is the frequency matrix where
    entry out[n] = x[n+1] - x[n]. This takes a R x T x L frequency array.
    """
    assert(freqs.ndim == 3)
    return np.diff(freqs, axis=1)


class TemporalFreqs(object):
    def __init__(self, freqs, samples, N=None, gintervals=None, swap=True):
        """

        Args:
            freqs: nsamples x nloci matrix
            samples: a list of tuples of (replicate, timepoint)
            gintervals: a GenomicIntervals object corresponding to the loci

        Notes:
        freqs is nsamples x nloci matrix. If samples includes replicates *and*
        timepoints, this will internally stored as a R x T x L tensor.

        """
        if not isinstance(freqs, np.ndarray) or freqs.ndim != 2:
            msg = ("argument 'freqs' must be a two dimensional "
                   "(S x L) numpy.ndarray.")
            raise ValueError(msg)
        if freqs.shape[0] > freqs.shape[1]:
            msg = (f"freqs matrix has more samples ({freqs.shape[0]}) than loci "
                   f"({freqs.shape[1]}) — did you supply the tranpose?")
            warnings.warn(msg)

        # Sort the samples, grouping by replicate and ordering timepoints
        # Then process sample, extracting replicates and timepoints,
        # and getting the number of unique replicates and timepoints
        samples, sorted_i = sort_samples(samples)
        replicates, timepoints, nreplicates, ntimepoints = process_samples(freqs, samples)
        self.samples = samples

        # process frequency matrix, turning into tensor
        self.freqs = reshape_freqs_to_tensor(freqs[sorted_i, :], nreplicates)
        self.N = None
        if N is not None:
            self.N = N[sorted_i, :].reshape((nreplicates, ntimepoints, -1))
            assert(self.N.shape == self.freqs.shape)
        self.swapped_alleles = None
        self.deltas = None
        self.cov = None
        self.covn = None
        self.gintervals = gintervals
        if swap:
            self.swap_alleles()

        # labels for the deltas
        # note that the timediffs are p[t] = p[t+1] - p[t]
        self.timediffs = np.tile(np.arange(1, self.ntimepoints), self.R)
        self.replicates = np.repeat(np.unique(replicates), self.T)

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    def load(self, file):
        with open(file, 'rb') as f:
            obj = pickle.load(f)
            return obj

    def __repr__(self):
        return (f"TemporalFreqs() object, {self.R} replicates x "
                 f"{self.T} timepoints x {self.L} loci")

    @property
    def ntimepoints(self):
        return self.freqs.shape[1]

    @property
    def L(self):
        return self.freqs.shape[2]

    @property
    def T(self):
        return self.freqs.shape[1]-1

    @property
    def R(self):
        return self.freqs.shape[0]

    def swap_alleles(self, force=False):
        "Swap the frequencies at randomly chosen loci."
        if not force and self.swapped_alleles is not None:
            raise ValueError("alleles already swapped.")
        self.freqs, self.swapped_alleles = swap_alleles(self.freqs)

    def calc_deltas(self, min_af=0.0, depth_limits=None):
        """
        This calculates the timepoint deltas matrix doing optional filtering on
        depth and minimum minor AF, which is the frequency matrix where entry
        out[n] = x[n+1] - x[n]. This is the notation used in Buffalo and Coop
        (2019) and numpy
        (https://docs.scipy.org/doc/numpy/reference/generated/numpy.diff.html).

        Differences are taken between adjacent timepoints. For an input tensor
        of R x T x L, a tensor of R x (T-1) x L is returned.
        """
        freqs = self.freqs
        freqs = filter_freqs(freqs, self.N, min_af, depth_limits)
        self.deltas = calc_deltas(freqs)
        return self.deltas


    def calc_covs_by_group(self, groups, min_af=0.0, depth_limits=None,
                           pairwise_complete=True, standardize=True,
                           bias=False, binomial_correction=True,
                           suppress_warnings=False):
        """
        Calculate covariances, grouping loci by the indices in groups.
        """
        freqs = filter_freqs(self.freqs, self.N, min_af, depth_limits)
        deltas = calc_deltas(freqs)
        covs, covns = covs_by_group(groups, freqs, deltas, self.N,
                                   pairwise_complete=pairwise_complete,
                                   standardize=standardize, bias=bias,
                                   binomial_correction=binomial_correction,
                                   suppress_warnings=suppress_warnings)
        self.tile_covs = covs
        self.tile_covns = covns
        return covs, covns

    def calc_covs(self, exclude_seqids=None,
                  min_af=0.0,
                  depth_limits=None,
                  pairwise_complete=True,
                  standardize=True, bias=False,
                  binomial_correction=True,
                  verbose=True,
                  suppress_warnings=False):
        """
        Calculate covariances; the heavy lifting is done by
        calc_temporal_replicate_covs().
        """
        freqs = filter_freqs(self.freqs, self.N, min_af, depth_limits, verbose=verbose)
        deltas = calc_deltas(freqs)
        N = self.N
        hets = calc_mean_hets(freqs, bias=False, N=N)
        #__import__('pdb').set_trace()
        # handle slicing out certin seqids
        loci_slice = slice_loci(self.gintervals.seqid,
                                exclude_seqids=exclude_seqids)
        Ns = None
        if binomial_correction:
            Ns = N[..., loci_slice]
        res = calc_temporal_replicate_covs(deltas[:, :, loci_slice],
                                           hets,
                                           N=Ns,
                                           freqs=freqs[..., loci_slice],
                                           pairwise_complete=pairwise_complete,
                                           standardize=standardize,
                                           bias=bias,
                                           binomial_correction=binomial_correction,
                                           suppress_warnings=suppress_warnings)
        self.cov, self.covn = res
        return self.cov


    def calc_covs_empirical_null(self, B=100, by_tile=False, verbose=True,
                                 exclude_seqids=None, min_af=0.0,
                                 depth_limits=None, pairwise_complete=True,
                                 sign_permute_blocks='seqid',
                                 binomial_correction=True,
                                 standardize=True, bias=False,
                                 suppress_warnings=False):
        """

        by_tile: whether to sign-flip by whole chromosomes (a conservative approach,
                 since no dependencies will be broken up) or by tile (less conservative).

        """
        all_covs = list()
        freqs = filter_freqs(self.freqs, self.N, min_af, depth_limits)
        N = self.N
        hets = calc_mean_hets(freqs, bias=False, N=N)
        # If we are doing things by tiles, we need to only include the loci in all
        # tiles, e.g. if some tiles drop certain loci at the ends.
        if by_tile:
            tile_loci = [locus for tile in self.tile_indices for locus in tile]
            loci_slice = slice_loci(self.gintervals.seqid, tile_loci, exclude_seqids)
            # need to change the seqid vector, which is for every locus, to only include
            # the loci in the tiles
            seqids = self.tile_seqids
        else:
            # just handle slicing out certin seqids
            loci_slice = slice_loci(self.gintervals.seqid, exclude_seqids=exclude_seqids)
            seqids = self.gintervals.seqid
        sliced_freqs = freqs[..., loci_slice]
        sliced_N = N[..., loci_slice]
        sliced_deltas = calc_deltas(freqs)[:, :, loci_slice]
        if verbose:
            B_range = tnrange(int(B))
        else:
            B_range = range(int(B))

        for b in B_range:
            # this permutes timepoints, and randomly flips the
            # sign for entire blocks loc loci, keeping the replicates
            # and loci unchanged. By specifying the chromosomes
            # as blocks, we allow differential flipping across chromosomes
            if sign_permute_blocks == 'tile':
                permuted_deltas = permute_delta_matrix(sliced_deltas, self.tile_ids, permute=False)
            # NOTE: I think we can do without the above, not artificially breaking up the
            # dependencies in the data.
            elif sign_permute_blocks == 'seqid':
                permuted_deltas = permute_delta_matrix(sliced_deltas, seqids, permute=False)
            else:
                raise ValueError("sign_permute_blocks must be either 'tile' or 'seqid'")

            if by_tile:
                covs, _ = covs_by_group(self.tile_indices, sliced_freqs, permuted_deltas,
                                        sliced_N, pairwise_complete=pairwise_complete,
                                        standardize=standardize, bias=bias,
                                        binomial_correction=binomial_correction,
                                        suppress_warnings=suppress_warnings)
            else:
                covs, _ = calc_temporal_replicate_covs(permuted_deltas,
                                                       hets, N=sliced_N,
                                                       freqs=sliced_freqs,
                                                       pairwise_complete=pairwise_complete,
                                                       standardize=standardize,
                                                       bias=bias,
                                                       binomial_correction=binomial_correction,
                                                       suppress_warnings=suppress_warnings)

            all_covs.append(covs)
        if by_tile:
            return reshape_empirical_null(all_covs, self.R, self.T)
        return all_covs

    def infer_gamma(self, ):
        """
        """
        T = self.T
        gammas = list()
        for i, tile_cov in enumerate(self.tile_covs):
            temp_tile_cov = stack_temporal_covariances(tile_cov, self.R, self.T)
            U, Sigma, V = np.linalg.svd(tile_cov[:, :, 0] - np.eye(T) / (2*10**log10N))
        np.eye(T)

    def F(self, exclude_seqids=None, min_af=0.0, depth_limits=None):
        """
        Returns var(p_t - p_0) / (p0(1-p0)) taken over loci.
        """
        loci_slice = slice_loci(self.gintervals.seqid, exclude_seqids=exclude_seqids)
        freqs = filter_freqs(self.freqs, min_af, depth_limits)
        pt, p0 = freqs[:, self.ntimepoints-1, loci_slice], freqs[:, 0, loci_slice]
        delta_p = pt - p0
        var = np.nanvar(delta_p, axis=1)
        norm = np.nanmean(p0 * (1-p0), axis=1)
        method = 'basic'
        # may be enabled later, TODO
        if method == 'JR07':
            z = (p0 + pt) / 2
            return np.nanmean((p0 - pt)**2  / (z - pt * p0), axis=1)
        return var/norm

    def _Fc(self, F, S=None, method='jonas16', *args, **kwargs):
        """
        S: number of individuals.

        Notes:
        A corrected F, based on Plan II Sampling (see Waples 1989) and the
        read-based binomial correction of Jónás et al (2016):
          Cj = 1/(2 * Sj) + 1/Rj - 1/(2*Sj*Rj)
        where Sj is the sample size in individuals, and Rj is the depth.
        The correction overally is based on Jónás et al (2016), eqn. 9.
        """
        if S is None:
            msg = ("method='jonas16' (from Jónás et al (2016)) requires S, "
                   "the number of individuals; proceeding with method='basic'")
            warnings.warn(msg)
            method = 'basic'
        Rt =  self.N.mean(axis=2)[:, self.ntimepoints-1]
        R0 =  self.N.mean(axis=2)[:, 0]
        if method == 'jonas16':
            Cj = lambda R: 1/(2 * S) + 1/R - 1/(2*S*R)
            C0 = Cj(R0)
            Ct = Cj(Rt)
            # experimental plan I:
            #return (F*(1 - 1/N) - C0 - Ct + 1/N) / (1-Ct)
            return (F - C0 - Ct) / (1-Ct)
        else:
           # a very simple correct: just subtract off noise due to sequencing variation
           return (F - 1/Rt - 1/R0)

    def Fc(self, S=None, *args, **kwargs):
        F = self.F(*args, **kwargs)
        return self._Fc(F=F, S=S, *args, **kwargs)

    def Ne(self, S=None, t=None, transform='simple', *args, **kwargs):
        """
        A basic Ne estimator, based on a corrected F (see TemporalFreqs.Fc).

        t: the number of generations elapsed during the observed
           frequencies (assuming even temporal sampling). By default
           this is None, and this forces it to assume sampling every generation.

        S: number of sampled individuals.
        """
        if t is None:
            t = self.ntimepoints
        if transform == 'log':
            return -t / (2*np.log(1-self.Fc(S, *args, **kwargs)))
        elif transform == 'simple':
            return t / (2*self.Fc(S, *args, **kwargs))
        else:
            raise ValueError("method must be 'log' or 'simple'")



    #def F(self, estimator=None, exclude_seqids=None):
    #    """
    #    Var(p_t - p_0) / E(p_0 (1 - p_0))
    #    where expectations for Var() and E() are taken over loci.
    #    """
    #    x = self.freqs[:, 0, :]
    #    y = self.freqs[:, self.ntimepoints-1, :]
    #    if estimator == 'KT71':
    #        ((x-y)**2) / (x*(1-x))
    #    denom = np.nanmean(p0*(1-p0), axis=1)
    #    return self.var(exclude_seqids) / denom

    # def write_freqs(self, filename, delim='\t', precision=8):
    #     cols = ['gen'] + list(map(str, self.loci))
    #     header = delim.join(cols)
    #     fmt = ['%d'] + [f'%.{int(precision)}e'] * len(self.loci)
    #     out = np.hstack((self.samples.reshape((-1, 1)),
    #                      self.freqs))
    #     np.savetxt(filename, out, delimiter=delim,
    #                comments='', fmt=fmt, header=header)


    # def write_covs(self, filename, long=True, delim='\t'):
    #     if not long:
    #         np.savetxt(filename, self.cov, delimiter=delim)
    #     else:
    #         rows, cols = np.indices(self.cov.shape)
    #         longmat = np.dstack((rows.reshape((-1, 1)),
    #                              cols.reshape((-1, 1)),
    #                              self.cov.reshape((-1, 1)))).squeeze()
    #         np.savetxt(filename, longmat, delimiter=delim,
    #                    comments='', fmt=('%d', '%d', '%.18e'),
    #                    header=delim.join(('row', 'col', 'cov')))


    def _G(self, end=None, abs=False, ignore_adjacent=False, double_offdiag=False):
        """
        Args:
            end: last timepoint to consider, useful for seeing
                 cumulative contributions
            abs: use absolute value of covariances
            ignore_adjacent: whether to ignore cov(Δp_{t}, Δp_{t+1}),
                             which is corrected for shared sampling noise

        ignore_adjacent is very conservative.
        """
        if self.cov is None:
            msg = "calculate covariances first with TemporalFreqs.calc_covs()"
            raise ValueError(msg)
        R, T = self.R, self.T
        covs = stack_temporal_covariances(self.cov, R, T, as_tensor=False)
        Gs = [calc_G(c, end, abs, ignore_adjacent, double_offdiag) for c in covs]
        return np.array(Gs)

    def G(self, end=None, abs=False, ignore_adjacent=False, double_offdiag=False):
        """
        Calculate G, the ratio of covariances to total variance. This is the ratio
        of the sum of off-diagonal elements to total sum of the covariance matrix.

        This returns an (T+1) x R matrix of cummulative G, up to time T.
        """
        G = np.stack([self._G(t) for t in np.arange(self.T+1)])
        return G


class TiledTemporalFreqs(TemporalFreqs):
    """
    A subclass of TemporalFreqs specifically for tiled EDA.

    Metadata:
     - self.tiles: a GenomicIntervals object containing the tile (window) ranges.
     - self.tile_indices: a ragged list, each element is a list of indices of
        the loci that belong in that tile.
     - self.tile_df: a convenience Pandas DataFrame containing the positions of the
        windows and any other metadata associated with them. This contains the midpoints
        of each tile, as well as the cummulative midpoint across all chromosomes.
     - self.tile_covs: the covariances for each tile, which is an (RxT) x (RxT)
        matrix.
     - self.tile_covns: the (biased) denominators of these covariances; the number of complete
        cases that went into the covariance calculation.
    """
    def __init__(self, tiles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_tiles(tiles)

    def _generate_ids(self):
        tile_ids = list()
        tile_seqids = list()
        for i, b in enumerate(self.tile_indices):
            tile_ids.extend([i]*len(b))
            tile_seqids.extend([self.tile_df['seqid'].values[i]]*len(b))
        self.tile_ids = np.array(tile_ids)
        self.tile_seqids = np.array(tile_seqids)

    def _seqid_tile_indices(self):
        seqid_groups = groupby(enumerate(self.gintervals.seqid), lambda x: x[1])
        return [[x[0] for x in v] for k, v in seqid_groups]

    def set_tiles(self, tiles):
        self.tiles = tiles
        self.tile_indices = self.tiles.nest_by_overlaps(self.gintervals, nest='index')
        self.seqid_tile_indices = self._seqid_tile_indices()
        self.tile_df = self.tiles.to_df(add_cummidpoint=True, add_midpoint=True)
        self._generate_ids()
        self.tile_covs = None
        self.tile_covns = None

    @property
    def ntiles(self):
        return len(self.tile_indices)

    def calc_covs_by_tile(self, keep_seqids=None, *args, **kwargs):
        """
        """
        indices = self.tile_indices
        seqid_tile_pairs = zip(self.tile_df['seqid'].values, self.tile_indices)
        if keep_seqids is None:
            keep_seqids = list(self.gintervals.keys())
        if isinstance(keep_seqids, list):
            keep_seqids = set(keep_seqids)
        else:
            raise ValueError("keep_seqids must be None (don't filter by seqid) or a list")
        indices = [indices for seqid, indices in seqid_tile_pairs if seqid in keep_seqids]
        return super().calc_covs_by_group(indices, *args, **kwargs)

    def avehet_by_tile(self, average=True):
        """
        Averages over all replicates and timepoints.
        """
        hets = []
        p = self.freqs.reshape((-1, self.freqs.shape[2]))
        ave_hets = np.nanmean((2 * p * (1-p)), axis=0)
        for indices in self.tile_indices:
            het = view_along_axis(ave_hets, indices, 0)
            if average:
                het = het.mean()
            hets.append(het)
        return hets

    def depth_by_tile(self, average=True):
        depth = []
        ave_N = self.N.mean(axis=(0, 1))
        for indices in self.tile_indices:
            n = view_along_axis(ave_N, indices, 0)
            if average:
                n = n.mean()
            depth.append(n)
        return depth

    def nloci_by_tile(self):
        return np.array([len(x) for x in self.tile_indices])

    def tile_cov_df(self, feather_file=None, verbose=True, temporal_only=True, *args, **kwargs):
        if verbose:
            print("calculating covariances...   ", end='')
        covs, covns = self.calc_covs_by_tile(*args, **kwargs)
        self.tile_covs = covs
        self.tile_covns = covns
        if verbose:
            print("done.")
        # Aggregate using some numpy tricks. This was previously done using
        # panda's DataFrame.groupby().agg() approach but this did not scale
        # well to large numbers of windows.
        if verbose:
            print("converting covariances to dataframes...    ", end='')
        cov_df = cov_matrices_to_dataframe(covs, self.R, self.T, temporal_only)
        if verbose:
            print("done.")
        # repeat the tile columns across each long covaraince matrix
        ntiles = len(covs)
        if temporal_only:
            ns = self.T
            each_repeated = (ns * (ns-1) / 2 + ns) * self.R
        else:
            ns = self.R * self.T
            each_repeated = ns * (ns-1) / 2 + ns
        self.tile_df['nloci'] = np.array([len(x) for x in self.tile_indices])
        rep_tile_df = self.tile_df.iloc[np.repeat(np.arange(ntiles), each_repeated)].reset_index()
        assert(rep_tile_df.shape[0] == cov_df.shape[0])
        cov_df = pd.concat([rep_tile_df, cov_df], axis=1)
        if feather_file is not None:
            cov_df.to_feather(feather_file)
            return
        return cov_df

    def correction_diagnostics(self, exclude_seqids=None, min_af=0.0,
                               depth_limits=None, offdiag_k=1, suppress_warnings=False):
        """
        Return a (DataFrame, models, ypreds) tuple full of diagonistic
        information that can be used in diagnostic plots (see
        cvtk.plots.correction_diagnostic_plot()) to assess the binomial correction procedure.
        This binomial correction affects the diagonal elements, removing binomial sampling noise,
        as well as the off-diagonal covariance elements, which share binomial smapling noise which
        adds a negative bias that is estimated and added back in. The dataframe contains the
        heterozygosity, the off-diagonal and diagonal elements of the covariance matrices, the depth,
        and other metadata. The models and ypreds elements are dictionaries with two values: True and False
        indicating whether the binomial correction has been applied to its values. The ypreds dictionary
        contains the linear regression predicted y values, while the models dictionary contains
        the actual regression fits.
        """
        tile_depth = self.depth_by_tile()
        dfs = list()
        models = dict()
        ypreds = dict()
        xpreds = dict()
        for use_correction in [True, False]:
            covs, covns = self.calc_covs_by_tile(min_af=min_af, depth_limits=depth_limits,
                                                 binomial_correction=use_correction,
                                                 suppress_warnings=suppress_warnings)
            offdiag = np.array([np.diag(c, k=offdiag_k).mean() for c in covs])
            diag = np.array([np.diag(c, k=0).mean() for c in covs])
            avehet = self.avehet_by_tile()
            seqid = self.tile_df['seqid'].values
            # load all this into a dataframe
            df = pd.DataFrame(dict(avehet=avehet, offdiag=offdiag, diag=diag,
                                   seqid=seqid, depth=tile_depth, correction=use_correction))
            if depth_limits is not None:
                df = df[(df['depth'] > depth_limits[0]) & (df['depth'] < depth_limits[1])]
            dfs.append(df)
            if exclude_seqids is not None:
                df = df[~df.seqid.isin(exclude_seqids)]
            diag_fit = smf.ols("diag ~ depth", df).fit()
            depth = df['depth'].dropna()
            diag_ypred = diag_fit.predict(dict(depth=depth))
            assert(len(diag_ypred.index) == len(depth))
            offdiag_fit = smf.ols("offdiag ~ depth", df).fit()
            offdiag_ypred = offdiag_fit.predict(dict(depth=depth))
            models[use_correction] = (diag_fit, offdiag_fit)
            xpreds[use_correction] = depth
            ypreds[use_correction] = (diag_ypred, offdiag_ypred)
        return (pd.concat(dfs, axis=0), models, xpreds, ypreds)

    def bootstrap_repcov(self, alpha, B, average_replicates=False,
                         keep_seqids=None, return_straps=False, min_af=0.0,
                         suppress_warnings=False,
                         depth_limits=None, percentile=False, binomial_correction=True):
        """
        This procedure bootstraps the replicate R x (R-1) / 2 replicate covariance matrices.

        There are R x (R-1) / 2 replicate covariance matrices, each T x T. If average_replicates
        is True, the R x (R-1) / 2 pairwise comparisons are averaged, leaving a T x T matrix.

        """
        covs, covns = self.calc_covs_by_tile(keep_seqids=keep_seqids,
                                             min_af=min_af, depth_limits=depth_limits,
                                             binomial_correction=binomial_correction,
                                             suppress_warnings=suppress_warnings)
        covs = np.stack([stack_replicate_covariances(c, self.R, self.T) for c in covs])
        #if self.T == 1:
        #    # only one time delta, try to squeeze replicate covs
        #    covs = covs.squeeze()
        # tile weights by number of loci
        keep_seqids = set(keep_seqids)
        indices_seqid_pairs = zip(self.tile_indices, self.tile_df['seqid'].values)
        weights = np.array([len(x) for x, seqid in indices_seqid_pairs if seqid in keep_seqids])
        weights = weights/weights.sum()

        # number of samples in resample
        N = covs.shape[0]
        straps = list()
        for b in np.arange(B):
            bidx = np.random.randint(0, N, size=N)
            # get the windows of the resampled indices
            mat = covs[bidx, ...]
            covs_masked = np.ma.masked_array(mat, np.isnan(mat))
            avecovs = np.ma.average(covs_masked, axis=0, weights=weights).data
            if average_replicates:
            # average across pairwise replicate matrices
                avecovs = avecovs.mean(axis=0)
            straps.append(avecovs)
        straps = np.stack(straps)
        That = np.mean(straps, axis=0)
        alpha = 100. * alpha  # because, numpy.
        qlower, qupper = (np.nanpercentile(straps, alpha/2, axis=0),
                          np.nanpercentile(straps, 100-alpha/2, axis=0))
        if return_straps:
            return straps
        if percentile:
            return qlower, That, qupper
        else:
            return 2*That - qupper, That, 2*That - qlower



    def bootstrap_tempcov(self, alpha, B, bootstrap_replicates=False,
                          replicate=None, average_replicates=False,
                          keep_seqids=None, return_straps=False, min_af=0.0,
                          depth_limits=None, percentile=False, binomial_correction=True,
                          suppress_warnings=False):
        """
        This procedure bootstraps the temporal sub-block covariance matrices (there are R of
        these, and each is TxT). Optionally, if bootstrap_replicates is True, the R replicates
        are resampled as well, and the covarainces are calculated for this sample as well. If
        replicate is supplied (an integer 0 ≤ replicate < R), then this procedure will return the
        bootstraps for this replicate only. If average_replicates is True, then the procedure will
        average across the replicates.

        This confidence interval returned is a pivotal CIs,
          C_l = 2 T - Q(1-α/2)
          C_u = 2 T - Q(α/2)
          where T is the estimator for the stastistic T, and α is the confidence level,
          and Q(x) is the empirical x percentile across the bootstraps.

        """
        covs, covns = self.calc_covs_by_tile(keep_seqids=keep_seqids,
                                             min_af=min_af, depth_limits=depth_limits,
                                             binomial_correction=binomial_correction,
                                             suppress_warnings=suppress_warnings)
        covs = np.stack([stack_temporal_covariances(c, self.R, self.T) for c in covs])
        if replicate is not None and bootstrap_replicates:
            msg = "cannot bootstrap on single replicate; set either bootstrap_replicates=False or replicate=None"
            raise ValueError(msg)
        if replicate is not None:
            covs = covs[:, :, :, replicate]
        # tile weights by number of loci
        keep_seqids = set(keep_seqids)
        indices_seqid_pairs = zip(self.tile_indices, self.tile_df['seqid'].values)
        weights = np.array([len(x) for x, seqid in indices_seqid_pairs if seqid in keep_seqids])
        weights = weights/weights.sum()

        # number of samples in resample
        N = covs.shape[0]
        straps = list()
        for b in np.arange(B):
            bidx = np.random.randint(0, N, size=N)
            # get the windows of the resampled indices
            mat = covs[bidx, ...]
            if bootstrap_replicates:
                assert(replicate is None)
                ridx = np.random.randint(0, self.R, size=self.R)
                mat = mat[:, :, :, ridx]
            covs_masked = np.ma.masked_array(mat, np.isnan(mat))
            avecovs = np.ma.average(covs_masked, axis=0, weights=weights).data
            if average_replicates:
                avecovs = avecovs.mean(axis=2)
            straps.append(avecovs)
        straps = np.stack(straps)
        That = np.mean(straps, axis=0)
        alpha = 100. * alpha  # because, numpy.
        qlower, qupper = np.nanpercentile(straps, alpha/2, axis=0), np.nanpercentile(straps, 100-alpha/2, axis=0)
        if return_straps:
            return straps
        if percentile:
            return qlower, That, qupper
        else:
            return 2*That - qupper, That, 2*That - qlower


    def bootstrap_cov(self, alpha, B, keep_seqids=None, return_straps=False,
                      min_af=0.0, depth_limits=None, percentile=False,
                      binomial_correction=True):
        """
        This procedure bootstraps the entire (RxT) x (RxT) covariance matrix using
        a block bootstrap procedure, where blocks are the genomic tiles.

        This confidence interval returned is a pivotal CIs,
          C_l = 2 T - Q(1-α/2)
          C_u = 2 T - Q(α/2)
          where T is the estimator for the stastistic T, and α is the confidence level,
          and Q(x) is the empirical x percentile across the bootstraps.
        """
        covs, covns = self.calc_covs_by_tile(keep_seqids=keep_seqids,
                                             min_af=min_af, depth_limits=depth_limits,
                                             binomial_correction=binomial_correction)
        covs = np.stack(covs)
        # tile weights by number of loci
        keep_seqids = set(keep_seqids)
        indices_seqid_pairs = zip(self.tile_indices, self.tile_df['seqid'])
        weights = np.array([len(x) for x, seqid in indices_seqid_pairs if seqid in keep_seqids])
        weights = weights/weights.sum()

        # number of samples in resample
        N = covs.shape[0]
        straps = list()
        for b in np.arange(B):
            bidx = np.random.randint(0, N, size=N)
            covs_masked = np.ma.masked_array(covs[bidx, :, :], np.isnan(covs[bidx, :, :]))
            straps.append(np.ma.average(covs_masked, axis=0, weights=weights).data)
        straps = np.stack(straps)
        That = np.mean(straps, axis=0)
        alpha = 100. * alpha  # because, numpy.
        qlower, qupper = (np.nanpercentile(straps, alpha/2, axis=0),
                          np.nanpercentile(straps, 100-alpha/2, axis=0))
        if return_straps:
            return straps
        if percentile:
            return qlower, That, qupper
        else:
            return 2*That - qupper, That, 2*That - qlower

#    def _bootstrap_rep_G(self, alpha, B, end=None, ignore_adjacent=False,
#                         return_straps=False, min_af=0.0, average_replicates=False,
#                         depth_limits=None, binomial_correction=True, suppress_warnings=False,
#                         percentile=False):
#        covs, covns = self.calc_covs_by_tile(min_af=min_af, depth_limits=depth_limits,
#                                             binomial_correction=binomial_correction,
#                                             suppress_warnings=suppress_warnings)
#        import pdb
#        rep_covs = np.stack([stack_replicate_covariances(c, self.R, self.T) for c in covs])
#        temp_covs = np.stack([stack_temporal_covariances(c, self.R, self.T) for c in covs])
#        # tile weights by number of loci
#        weights = np.array([len(x) for x in self.tile_indices])
#        weights = weights/weights.sum()
#
#        # number of samples in resample
#        N = len(self.tile_indices)
#        straps = list()
#        for b in np.arange(B):
#            bidx = np.random.randint(0, N, size=N)
#            rep_covs_masked = np.ma.masked_array(rep_covs[bidx, :, :], np.isnan(covs[bidx, :, :]))
#            temp_covs_masked = np.ma.masked_array(temp_covs[bidx, :, :], np.isnan(covs[bidx, :, :]))
#            rep_avecov = np.ma.average(rep_covs_masked, axis=0, weights=weights).data
#            temp_avecov = np.ma.average(temp_covs_masked, axis=0, weights=weights).data
#            repcovs = list()
#            # iterate through all the replicates, calculating G for each one.
#            for rep in np.arange(self.R):
#                G = calc_rep_G(rep_avecov[:, :, rep], temp_avecov[:,:,rep],
#                        end=end, ignore_adjacent=ignore_adjacent)
#                repcovs.append(G)
#                pdb.set_trace()
#            repcovs = np.stack(repcovs)
#            if average_replicates:
#                repcovs = np.mean(repcovs)
#            straps.append(repcovs)
#        That = np.mean(straps, axis=0)
#        alpha = 100. * alpha  # because, numpy.
#        qlower, qupper = (np.nanpercentile(straps, alpha/2, axis=0),
#                          np.nanpercentile(straps, 100-alpha/2, axis=0))
#        if return_straps:
#            return straps
#        if percentile:
#            return qlower, That, qupper
#        else:
#            return 2*That - qupper, That, 2*That - qlower


    def _bootstrap_G(self, alpha, B, end=None, ignore_adjacent=False,
                    return_straps=False, min_af=0.0, average_replicates=False,
                    depth_limits=None, binomial_correction=True,
                    suppress_warnings=False, percentile=False):
        # TODO:
        covs, covns = self.calc_covs_by_tile(min_af=min_af, depth_limits=depth_limits,
                                             binomial_correction=binomial_correction,
                                             suppress_warnings=suppress_warnings)
        covs = np.stack([stack_temporal_covariances(c, self.R, self.T) for c in covs])
        # tile weights by number of loci
        weights = np.array([len(x) for x in self.tile_indices])
        weights = weights/weights.sum()

        # number of samples in resample
        N = len(self.tile_indices)
        straps = list()
        for b in np.arange(B):
            bidx = np.random.randint(0, N, size=N)
            covs_masked = np.ma.masked_array(covs[bidx, :, :], np.isnan(covs[bidx, :, :]))
            avecov = np.ma.average(covs_masked, axis=0, weights=weights).data
            repcovs = list()
            # iterate through all the replicates, calculating G for each one.
            for rep in np.arange(self.R):
                repcovs.append(calc_G(avecov[:, :, rep], end=end, ignore_adjacent=ignore_adjacent))
            repcovs = np.stack(repcovs)
            if average_replicates:
                repcovs = np.mean(repcovs)
            straps.append(repcovs)
        That = np.mean(straps, axis=0)
        alpha = 100. * alpha  # because, numpy.
        qlower, qupper = (np.nanpercentile(straps, alpha/2, axis=0),
                          np.nanpercentile(straps, 100-alpha/2, axis=0))
        if return_straps:
            return straps
        if percentile:
            return qlower, That, qupper
        else:
            return 2*That - qupper, That, 2*That - qlower

    def bootstrap_G(self, *args, **kwargs):
        """
        Like TemporalFreqs.G(), this calls the TemporalFreqs._bootstrap_G() method for each
        timepoint, constructing a block bootstrapped G, blocked by tile.

        This returns a 3 x (T+1) x R array, where the first dimension is lower-CI, mean, upper CI of
        G.
        """
        G = np.stack([self._bootstrap_G(*args, end=t, **kwargs) for t in np.arange(self.T+1)])
        return G.swapaxes(0, 1)




    def G_by_tile(self, end=None, abs=False, ignore_adjacent=True, *args, **kwargs):
        covs, covns = self.calc_covs_by_tile(*args, **kwargs)
        covs_temp = [[calc_G(rc, end, abs, ignore_adjacent) for rc
                      in stack_temporal_covariances(c, self.R, self.T, as_tensor=False)] for c in covs]
        return np.array(covs_temp)

