import itertools
import numpy as np

from cvtk.utils import swap_alleles


def neut_sfs(nloci, n):
    probs = 1/np.arange(1, n)
    probs /= sum(probs)
    return np.random.choice(np.arange(1, n)/n, nloci, replace=True, p=probs)

def wright_fisher(N, L, ngens, p0=None, swap=True):
    # returns a matrix of ngens x L
    # create initial frequencies
    if p0 is None:
        p0 = neut_sfs(L, 2*N)
    else:
        p0 = np.repeat(p0, L)

    # simulate frequency trajectories
    freqs = [p0]
    for gen in range(1, ngens):
        new_freqs = np.random.binomial(2*N, freqs[gen-1], size=L) / (2*N)
        freqs.append(new_freqs)
    freqs_mat = np.stack(freqs)
    if swap:
        freqs_mat, swaps = swap_alleles(freqs_mat)
    return freqs_mat

def sample_depth(freqs, depth, diploids=None, poisson=False):
    if poisson:
        depth = np.random.poisson(depth, freqs.shape)
    else:
        depth = np.broadcast_to(depth, freqs.shape)
    if diploids is not None:
        freqs = np.random.binomial(2*diploids, freqs, freqs.shape) / (2*diploids)
    return np.random.binomial(depth, freqs, freqs.shape), depth


def wright_fisher_sample(N, L, ngens, depth, poisson=False,
                         diploids=None, p0=None, swap=True,
                         *args, **kwargs):
    freqs = wright_fisher(N, L, ngens, p0=p0, swap=swap)
    counts, depth = sample_depth(freqs, depth=depth, diploids=diploids, poisson=poisson)
    sample_freqs = counts / depth
    return freqs, sample_freqs, counts, depth

def param_grid(**params):
    keys, values = zip(*params.items())
    return [dict(zip(keys, vals)) for vals in itertools.product(*values)]

