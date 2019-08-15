from warnings import warn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import allel
import numpy as np
import pandas as pd


class FreqPCA(object):
    """
    Given a frequency matrix (nloci x nsamples) calculate the PCs of the samples.
    """

    def __init__(self, mat, n_components=None):
        """

        Given an nloci x nsamples matrix, using a denominator similar to that of
        Patterson et al. (2006). This is originally intended for a PCA of genotypes,
        not frequencies, so we vary it to be:

        (p_ij - μ_j) / sqrt(μ_j (1 - μ_j))

        for pop i, locus j, μ_j is mean frequency for locus j.

        mat: nloci x ninds
        
        """
        if n_components is None:
            n_components = mat.shape[1]
        nloci = mat.shape[0]
        p_bar = mat.mean(axis=1)
        denom = np.sqrt(p_bar * (1-p_bar))
        keep = denom > 0
        nvalid = keep.sum()
        nremoved = nloci - nvalid
        if nremoved > 0:
            perc = np.round(nremoved / nloci, 2) * 100
            warn(f"removing {nremoved} loci ({perc}%) due to NaN/non-finite values")
        mat = mat[keep, :]
        self.denom = denom[keep]
        self.centered = (mat.T - p_bar).T
        self.pca = PCA(n_components=n_components)
        self.X = (self.centered.T / self.denom).T
        self.X = self.centered.T

    @property
    def pcs(self):
        return self.pca.fit_transform(self.X)

    @property
    def explained_variance(self):
        return self.pca.explained_variance_

    def to_df(self):
        pcs = self.pcs
        pc_labels = [f"pc{i}" for i in np.arange(1, pcs.shape[1]+1)]
        pc_df = pd.DataFrame(data=pcs, columns=pc_labels)
        return pc_df

