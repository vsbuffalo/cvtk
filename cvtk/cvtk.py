## cvtk.py


from itertools import groupby
import numpy as np
import pandas as pd

from cvtk.utils import sort_samples, swap_alleles, reshape_matrix
from cvtk.utils import process_samples, view_along_axis
from cvtk.cov import temporal_cov, covs_by_group, calc_hets
from cvtk.cov import stack_temporal_covs_by_group
from cvtk.bootstrap import block_bootstrap_temporal_covs
from cvtk.G import calc_G, block_estimate_G
from cvtk.diagnostics import calc_diagnostics

class TemporalFreqs(object):
    """
    """
    def __init__(self, freqs, samples, depths=None, diploids=None,
                 gintervals=None, swap=True):

        if freqs.shape != depths.shape:
            msg = ("frequency matrix 'freqs' must have same shape as"
                   "matrix of sequencing depths 'depths'")
            raise ValueError(msg)

        if freqs.shape[0] > freqs.shape[1]:
            msg = (f"freqs matrix has more samples ({freqs.shape[0]}) than loci"
                   f"({freqs.shape[1]}) — did you supply the tranpose?")
            warnings.warn(msg)

        # Sort the samples, grouping by replicate and ordering timepoints
        # Then process sample, extracting replicates and timepoints,
        # and getting the number of unique replicates and timepoints
        samples, sorted_i = sort_samples(samples)
        replicates, timepoints, nreplicates, ntimepoints = process_samples(freqs, samples)
        self.samples = samples

        # process frequency matrix, turning into tensor
        self.freqs = reshape_matrix(freqs[sorted_i, :], nreplicates)
        self.depths = reshape_matrix(depths[sorted_i, :], nreplicates)

        self.swapped_alleles = None
        if swap:
            self.freqs, self.swapped_alleles = swap_alleles(self.freqs)

        # TODO: reshape diploids according to samples
        self.diploids = diploids
        self.gintervals = gintervals

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

    def calc_covs(self, exlude_seqs=None, bias_correction=True):
        """
        Calculate the genome-wide temporal-replicate variance-covariance matrix.
        """
        return temporal_cov(self.freqs, self.depths, self.diploids,
                            bias_correction=bias_correction)

    def calc_covs_by_group(self, groups, bias_correction=True):
        """
        Calculate covariances, grouping loci by the indices in groups.
        """
        covs = covs_by_group(groups, self.freqs, depths=self.depths,
                             diploids=self.diploids,
                             bias_correction=bias_correction)
        self.tile_covs = covs
        return covs



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
    """
    def __init__(self, tiles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_tiles(tiles)

    def _generate_ids(self):
        """
        Build two attributes: TiledTemporalFreqs.tile_ids and 
        TiledTemporalFreqs.tile_seqids. Tiles are stored in the 
        TiledTemporalFreqs.tile_indices atribute, which is a raggest list 
        with the list containing the tile lists, and each tile lists contains
        the indices for the variants. This generates the TiledTemporalFreqs.tile_ids
        and TiledTemporalFreqs.tile_seqids arrays, which indicate which number tile 
        each SNP is (their length is equal to the total number of variants).
        """
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

    @property
    def ntiles(self):
        return len(self.tile_indices)

    def calc_covs_by_tile(self, *args, **kwargs):
        indices = self.tile_indices
        return super().calc_covs_by_group(indices, *args, **kwargs)

    def depth_by_tile(self, average=True):
        depth = []
        ave_depth = self.depths.mean(axis=(0, 1))
        for indices in self.tile_indices:
            n = view_along_axis(ave_depth, indices, 0)
            if average:
                n = n.mean()
            depth.append(n)
        return depth

    def calc_het_by_tile(self, average=True, bias=False):
        """
        Averages over all replicates and timepoints.
        """
        tile_hets = []
        hets = calc_hets(self.freqs, self.depths, self.diploids, bias=bias)
        for indices in self.tile_indices:
            het = view_along_axis(hets, indices, 0)
            if average:
                het = het.mean()
            tile_hets.append(het)
        return tile_hets



    def correction_diagnostics(self, exclude_seqids=None, offdiag_k=1):
        """
        Return a (DataFrame, models, ypreds) tuple full of diagnostic information that can be 
        used in diagnostic plots (see
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
        dfs = list()
        models = dict()
        ypreds = dict()
        xpreds = dict()
        # data needed for diagnostics, but not affected by the bias correction
        tile_depths = self.depth_by_tile()
        mean_hets = self.calc_het_by_tile()
        seqids = self.tile_df['seqid'].values
        for use_correction in [True, False]:
            covs = self.calc_covs_by_tile(bias_correction=use_correction)
            res = calc_diagnostics(covs, mean_hets, seqids, tile_depths, exclude_seqids=exclude_seqids)
            models[use_correction], xpreds[use_correction], ypreds[use_correction], df = res
            df['correction'] = use_correction
            dfs.append(df)
        return pd.concat(dfs, axis=0), models, xpreds, ypreds

    def bootstrap_temporal_covs(self, B, alpha=0.05, bootstrap_replicates=False,
                                replicate=None, average_replicates=False, 
                                keep_seqids=None, return_straps=False,
                                ci_method='pivot', **kwargs):
        """
        Wrapper around block_bootstrap_temporal_covs().
        Params: 
           - B: number of bootstraps
           - alpha: α level
           - bootstrap_replicates: whether the R replicates are resampled as well, and 
              covariance is averaged over these replicates.
           - replicate: only bootstrap the covariances for a single replicate (cannot be used 
              with bootstrap_replicates).
           - average_replicates: whether to average across all replicates.
           - keep_seqids: which seqids to include in bootstrap; if None, all are used.
           - return_straps: whether to return the actual bootstrap vectors.
           - ci_method: 'pivot' or 'percentile'
           - **kwargs: based to calc_covs_by_tile()
 
        """
        covs = stack_temporal_covs_by_group(self.calc_covs_by_tile(**kwargs), self.R, self.T)
        return block_bootstrap_temporal_covs(covs, 
                     block_indices=self.tile_indices, block_seqids=self.tile_df['seqid'],
                     B=B, alpha=alpha, 
                     bootstrap_replicates=bootstrap_replicates, 
                     average_replicates=average_replicates,
                     keep_seqids=keep_seqids, return_straps=return_straps, 
                     ci_method=ci_method)


    def _bootstrap_Gs(self, B, end=None, abs=False, alpha=0.05, 
                               bootstrap_replicates=False,
                               replicate=None, average_replicates=False, 
                               keep_seqids=None, return_straps=False,
                               ci_method='pivot', **kwargs):
        """
        Wrapper around block_bootstrap_temporal_covs(), with the estimator function
        block_estimate_G().
        Params: 
           - B: number of bootstraps
           - alpha: α level
           - bootstrap_replicates: whether the R replicates are resampled as well, and 
              covariance is averaged over these replicates.
           - replicate: only bootstrap the covariances for a single replicate (cannot be used 
              with bootstrap_replicates).
           - average_replicates: whether to average across all replicates.
           - keep_seqids: which seqids to include in bootstrap; if None, all are used.
           - return_straps: whether to return the actual bootstrap vectors.
           - ci_method: 'pivot' or 'percentile'
           - **kwargs: based to calc_covs_by_tile()
 
        """
        covs = stack_temporal_covs_by_group(self.calc_covs_by_tile(**kwargs), self.R, self.T)
        return block_bootstrap_temporal_covs(covs, 
                     block_indices=self.tile_indices, block_seqids=self.tile_df['seqid'],
                     B=B, 
                     estimator=block_estimate_G,
                     alpha=alpha, 
                     bootstrap_replicates=bootstrap_replicates, 
                     average_replicates=average_replicates,
                     keep_seqids=keep_seqids, return_straps=return_straps, 
                     ci_method=ci_method)


    def bootstrap_Gs(self, *args, **kwargs):
        """
        Like TemporalFreqs.G(), this calls the TemporalFreqs._bootstrap_G() method for each
        timepoint, constructing a block bootstrapped G, blocked by tile.

        This returns a 3 x (T+1) x R array, where the first dimension is lower-CI, mean, upper CI of
        G.
        """
        G = np.stack([self._bootstrap_Gs(*args, end=t, **kwargs) for t in np.arange(self.T+1)])
        return G.swapaxes(0, 1)


