## cvtk.py

from cvtk.utils import sort_samples, swap_alleles, reshape_matrix
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
            self.freqs, self.swapped_alleles = swap_alleles(freqs)

        # TODO: reshape diploids according to samples
        self.diploids = diploids
        self.gintervals = gintervals


    def cov(self):
        """
        Calculate the genome-wide temporal-replicate variance-covariance matrix.
        """
        return temporal_cov(self.freqs, self.depths, self.diploids)



