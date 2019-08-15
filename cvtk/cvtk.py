## cvtk.py

from cvtk.utils import sort_samples, swap_alleles, reshape_matrix
from cvtk.utils import process_samples
from cvtk.cov import temporal_cov

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

    def calc_covs(self, exlude_seqs=None):
        """
        Calculate the genome-wide temporal-replicate variance-covariance matrix.
        """
        return temporal_cov(self.freqs, self.depths, self.diploids)

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




