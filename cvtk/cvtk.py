## cvtk.py


from itertools import groupby
import numpy as np
import warnings
import pandas as pd
from tqdm import tnrange

from cvtk.utils import sort_samples, swap_alleles, reshape_matrix
from cvtk.utils import process_samples, view_along_axis, validate_diploids
from cvtk.cov import stack_replicate_covs_by_group, stack_temporal_covariances
from cvtk.cov import temporal_replicate_cov, cov_by_group, calc_hets
from cvtk.cov import total_variance, var_by_group
from cvtk.cov import stack_temporal_covs_by_group, stack_replicate_covs_by_group
from cvtk.cov import replicate_block_matrix_indices
from cvtk.bootstrap import block_bootstrap_ratio_averages, cov_estimator
from cvtk.bootstrap import block_bootstrap
from cvtk.G import calc_G, G_estimator, convergence_corr_by_group
from cvtk.G import convergence_corr_numerator, convergence_corr_denominator, convergence_corr
from cvtk.G import convergence_corr_from_freqs
from cvtk.diagnostics import calc_diagnostics
from cvtk.empirical_null import calc_covs_empirical_null

class TemporalFreqs(object):
    """
    """
    def __init__(self, freqs, samples, depths=None, diploids=None,
                 gintervals=None, swap=True, share_first=False):

        if depths is not None and freqs.shape != depths.shape:
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
        self.share_first = share_first

        # process frequency matrix, turning into tensor
        self.freqs = reshape_matrix(freqs[sorted_i, :], nreplicates)
        self.depths = None
        if depths is not None:
            # depths can be stored much more efficiently as a uint16
            assert(np.all(np.nanmax(depths) < np.iinfo(np.uint16).max))
            depths = depths.astype('uint16')
            self.depths = reshape_matrix(depths[sorted_i, ...], nreplicates)

        self.diploids = None
        if diploids is not None:
            if isinstance(diploids, np.ndarray) and len(diploids) > 1:
                #assert(diploids.ndim == 1)
                try:
                    diploids = diploids[sorted_i, ...]
                except:
                    msg = ("diploids must be single integer or array of "
                           f"size nreplicates*ntimepoints ({nreplicates*ntimepoints}), "
                           f"supplied size: {len(diploids)}")
                    raise ValueError(msg)
            self.diploids = validate_diploids(diploids, nreplicates, ntimepoints)
            #assert(self.diploids.shape == (nreplicates, ntimepoints, 1))

        self.swapped_alleles = None
        if swap:
            self.freqs, self.swapped_alleles = swap_alleles(self.freqs)

        # TODO: reshape diploids according to samples
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

    def calc_cov(self, keep_seqids=None, bias_correction=True,
                 standardize=True, use_masked=False, product_only=False):
        """
        Calculate the genome-wide temporal-replicate variance-covariance matrix.

        If keep_seqids=None, all seqids are used in the calculation of the covariance.
        Otherwise, if it's a list of seqids, only these are used.
        """
        freqs, depths = self.freqs, self.depths
        if keep_seqids is not None:
            assert(isinstance(keep_seqids, (tuple, list)))
            idx = np.array([i for i, seqid in enumerate(self.gintervals.seqid)
                            if seqid in keep_seqids])
            freqs, depths = freqs[..., idx], depths[..., idx]
        return temporal_replicate_cov(freqs, depths, self.diploids,
                                      bias_correction=bias_correction,
                                      share_first=self.share_first,
                                      product_only=product_only,
                                      standardize=standardize, use_masked=use_masked)

    def convergence_corr(self, subset=None, bias_correction=True, **kwargs):
        gw_covs = self.calc_cov(standardize=False,
                                bias_correction=bias_correction, **kwargs)
        if subset is not None:
            rows, cols = replicate_block_matrix_indices(self.R, self.T)
            rows_keep, cols_keep = np.isin(rows, subset), np.isin(cols, subset)
            dim = self.T * len(subset)
            gw_covs = gw_covs[np.logical_and(rows_keep, cols_keep)].reshape((dim, dim))

        R = self.R if subset is None else len(subset)
        return convergence_corr(gw_covs, R, self.T)

    def convergence_corr_by_group(self, groups, bias_correction=True):
        covs = self.calc_cov_by_group(groups, standardize=False, bias_correction=bias_correction)
        return convergence_corr_by_group(covs, self.R, self.T)


    def calc_cov_by_group(self, groups, keep_seqids=None, bias_correction=True,
                           standardize=True, use_masked=False, return_ratio_parts=False,
                           progress_bar=False):
        """
        Calculate covariances, grouping loci by the indices in groups.
        """
        freqs, depths = self.freqs, self.depths
        if keep_seqids is not None:
            assert(isinstance(keep_seqids, (tuple, list)))
            idx = np.array([i for i, seqid in enumerate(self.gintervals.seqid)
                            if seqid in keep_seqids])
            freqs, depths = freqs[..., idx], depths[..., idx]
        res = cov_by_group(groups, freqs,
                            depths=depths, diploids=self.diploids,
                            standardize=standardize,
                            use_masked=use_masked, share_first=self.share_first,
                            return_ratio_parts=return_ratio_parts,
                            bias_correction=bias_correction, progress_bar=progress_bar)
        if return_ratio_parts:
            covs, het_denoms = res
            return covs, het_denoms
        else:
            return res

    def calc_var(self, t=None, standardize=True, bias_correction=True):
        return total_variance(self.freqs, self.depths, self.diploids, t=t,
                              standardize=standardize, bias_correction=bias_correction)

    def calc_var_by_group(self, groups, t=None, standardize=True, bias_correction=True):
        return var_by_group(groups, freqs=self.freqs, depths=self.depths, diploids=self.diploids,
                            t=t, standardize=standardize, bias_correction=bias_correction)

    def calc_G(self, average_replicates=False, abs=False,
               product_only=False,
               use_masked=False):
        covs = self.calc_cov(standardize=False,
                             product_only=product_only,
                             use_masked=use_masked)
        # calculate the total variances for different ts
        vars = []
        for t in np.arange(1, self.T+1):
            vars.append(np.stack(self.calc_var(t=t, standardize=False)))
        total_vars = np.stack(vars, axis=0)
        temp_covs = stack_temporal_covariances(covs, self.R, self.T)
        G = G_estimator(temp_covs, total_vars,
                        average_replicates=average_replicates, abs=abs)
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

    def calc_cov_by_tile(self, *args, **kwargs):
        indices = self.tile_indices
        return super().calc_cov_by_group(indices, *args, **kwargs)

    def calc_var_by_tile(self, *args, **kwargs):
        indices = self.tile_indices
        return super().calc_var_by_group(indices, *args, **kwargs)


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
            #het = view_along_axis(hets, indices, 0)
            het = hets[..., indices]
            if average:
                het = het.mean()
            tile_hets.append(het)
        return tile_hets



    def correction_diagnostics(self, exclude_seqids=None, offdiag_k=1,
                               depth_limits=None, **kwargs):
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
            covs = self.calc_cov_by_tile(bias_correction=use_correction, **kwargs)
            res = calc_diagnostics(covs, mean_hets, seqids, tile_depths,
                                   exclude_seqids=exclude_seqids,
                                   depth_limits=depth_limits)
            models[use_correction], xpreds[use_correction], ypreds[use_correction], df = res
            df['correction'] = use_correction
            dfs.append(df)
        return pd.concat(dfs, axis=0), models, xpreds, ypreds

    def bootstrap_cov(self, B, alpha=0.05, keep_seqids=None, progress_bar=False,
                      average_replicates=True, return_straps=False, ci_method='pivot',
                      **kwargs):
        """
        Bootstrap whole covaraince matrix.
        Params:
           - B: number of bootstraps
           - alpha: α level
           - keep_seqids: which seqids to include in bootstrap; if None, all are used.
           - return_straps: whether to return the actual bootstrap vectors.
           - ci_method: 'pivot' or 'percentile'
           - **kwargs: passed to calc_cov() and calc_cov_by_tile()

        """
        # our statistic:
        cov = self.calc_cov(keep_seqids=keep_seqids, **kwargs)
        if average_replicates:
            cov = np.nanmean(stack_temporal_covariances(cov, self.R, self.T), axis=2)
        covs, het_denoms = self.calc_cov_by_tile(return_ratio_parts=True,
                                                 keep_seqids=keep_seqids, **kwargs)
        covs, het_denoms = np.stack(covs), np.stack(het_denoms)
        return block_bootstrap_ratio_averages(covs, het_denoms,
                                              block_indices=self.tile_indices,
                                              block_seqids=self.tile_df['seqid'].values,
                                              estimator=cov_estimator,
                                              statistic=cov,
                                              B=B, alpha=alpha,
                                              progress_bar=progress_bar,
                                              keep_seqids=keep_seqids, return_straps=return_straps,
                                              ci_method=ci_method,
                                              # kwargs passed directly to cov_estimator
                                              average_replicates=average_replicates,
                                              R=self.R, T=self.T)

    def convergence_corr_by_tile(self, bias_correction=True):
        return self.convergence_corr_by_group(self.tile_indices,
                                              bias_correction=bias_correction)


    def bootstrap_convergence_corr(self, B, subset=None, alpha=0.05, keep_seqids=None,
                                   ci_method='pivot', progress_bar=False,
                                   bias_correction=True, ratio_of_averages=True,
                                   return_straps=False):
        """
        subset: the indices of the samples to calculate this on; if subset=None, this
                uses all.

        Bootstrap the convergence correlation. By default this is calculated as a ratio of
        averages,

        E_{A≠B} cov(Δp_{t,A}, Δp_{t,B})
        -----------------------
        E_{A≠B} sqrt(var(Δp_{t,A}) var(Δp_{t,B}))

        But if ratio_of_averages = False, blocks of frequencies are sampled, the covariance calculated,
        and then the convergence correlation is calculated on those.
        """
        # our statistic:
        conv_corr = self.convergence_corr(subset=subset,
                                          keep_seqids=keep_seqids,
                                          bias_correction=bias_correction)

        if not ratio_of_averages:
            return block_bootstrap(self.freqs, B=B, block_indices=self.tile_indices,
                                   block_seqids=self.tile_df['seqid'].values,
                                   alpha=alpha, keep_seqids=keep_seqids,
                                   progress_bar=progress_bar,
                                   estimator=convergence_corr_from_freqs,
                                   statistic=conv_corr,
                                   standardize=False, bias_correction=bias_correction)

        covs = self.calc_cov_by_tile(bias_correction=bias_correction,
                                     keep_seqids=keep_seqids, standardize=False)
        if subset is not None:
            rows, cols = replicate_block_matrix_indices(self.R, self.T)
            rows_keep, cols_keep = np.isin(rows, subset), np.isin(cols, subset)
            dim = self.T * len(subset)
            covs = [cov[np.logical_and(rows_keep, cols_keep)].reshape((dim, dim)) for cov in covs]


        R = self.R if subset is None else len(subset)
        replicate_covs = stack_replicate_covs_by_group(covs, R, self.T, upper_only=False)
        temporal_covs = stack_temporal_covs_by_group(covs, R, self.T)

        nums, denoms = (convergence_corr_numerator(replicate_covs),
                        convergence_corr_denominator(temporal_covs))
        return block_bootstrap_ratio_averages(nums, denoms,
                     block_indices=self.tile_indices,
                     block_seqids=self.tile_df['seqid'].values,
                     estimator=np.divide,
                     B=B,
                     statistic=conv_corr,
                     alpha=alpha,
                     keep_seqids=keep_seqids,
                     return_straps=return_straps,
                     ci_method=ci_method, progress_bar=progress_bar)


    def bootstrap_G(self, B, abs=False, alpha=0.05, keep_seqids=None,
                    use_masked=False,
                    average_replicates=False, ci_method='pivot',
                    progress_bar=False, **kwargs):
        """
        """
        # calc our statistic:
        G = self.calc_G(average_replicates=average_replicates,
                        use_masked=use_masked, abs=abs)
        vars = list()
        for t in np.arange(1, self.T+1):
            vars.append(np.stack(self.calc_var_by_tile(t=t, standardize=False)))
        total_vars = np.stack(vars, axis=1)
        tile_covs = self.calc_cov_by_tile(standardize=False,
                                          use_masked=use_masked,
                                          keep_seqids=keep_seqids)
        covs = stack_temporal_covs_by_group(tile_covs, self.R, self.T)
        return block_bootstrap_ratio_averages(covs, total_vars,
                     block_indices=self.tile_indices,
                     block_seqids=self.tile_df['seqid'].values,
                     estimator=G_estimator,
                     B=B,
                     statistic=G,
                     alpha=alpha,
                     keep_seqids=keep_seqids, return_straps=False,
                     ci_method=ci_method, progress_bar=progress_bar,
                     # kwargs passed directly to G_estimate
                     average_replicates=average_replicates,
                     abs=abs)

    def calc_empirical_null(self, B=100, exclude_seqs=None,
                       sign_permute_blocks='tile',
                       by_tile=False,
                       use_masked=False,
                       bias_correction=False, progress_bar=False):

        return calc_covs_empirical_null(self.freqs,
                                        tile_indices=self.tile_indices,
                                        tile_seqids=self.tile_df['seqid'],
                                        tile_ids=self.tile_ids,
                                        gintervals=self.gintervals,
                                        B=B,
                                        depths=self.depths,
                                        use_masked=use_masked,
                                        diploids=self.diploids,
                                        by_tile=by_tile,
                                        exclude_seqids=exclude_seqs,
                                        sign_permute_blocks=sign_permute_blocks,
                                        bias_correction=bias_correction,
                                        progress_bar=progress_bar)

