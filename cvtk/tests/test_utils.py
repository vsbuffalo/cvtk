import pytest
import numpy as np

from cvtk.utils import swap_alleles

def test_swap_alleles_2d():
    np.random.seed(0)
    freqs = np.random.uniform(0, 1, (10, 12))
    swapped_freqs, swaps = swap_alleles(freqs)
    # if we swap back, are they the same?
    assert(np.all(np.abs(swapped_freqs - swaps) == freqs))

def test_swap_alleles_3d():
    np.random.seed(0)
    freqs = np.random.uniform(0, 1, (5, 10, 12))
    swapped_freqs, swaps = swap_alleles(freqs)
    # if we swap back, are they the same?
    assert(np.all(np.abs(swapped_freqs - swaps) == freqs))

