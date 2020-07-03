import os
import re
import glob
import pickle
from operator import itemgetter
import random
from multiprocessing import Pool
from collections import defaultdict
import pandas as pd
import numpy as np
from matplotlib.patches import Polygon
import statsmodels.api as sm


from cvtk.cvtk import TemporalFreqs, TiledTemporalFreqs
from cvtk.cov import stack_temporal_covariances
from cvtk.gintervals import GenomicIntervals
import cvtk.slimfile as sf

def freqs_to_covmat(frqs, times,
                    pops=None, fixed_to_nan=False,
                    product_only=False,
                    return_object=False, loci=None):
    """
    Take a frequency file, load it into cvtk, and convert to
    a tuple of covariances, Gs, and the object itself.

    Mainly used for the split population simulation results.
    """
    use_masked = fixed_to_nan
    samples = [(pop, time) for pop, time in zip(pops, np.concatenate((times, times)))]
    freqmat = frqs.copy()
    if fixed_to_nan:
        freqmat[np.logical_or(freqmat == 0., freqmat == 1.)] = np.nan
    if return_object:
        # make a ginterval object for loci
        gi = GenomicIntervals()
        for locus in sorted(loci):
            # assumes this is out of SlimFile processing funcs,
            # positions is list.
            gi.append('1', locus)  # only one chrom in sims
        d = TemporalFreqs(freqmat, samples, gintervals=gi)
        return d
    d = TemporalFreqs(freqmat, samples)
    covs = d.calc_cov(use_masked=use_masked, product_only=product_only)
    G = d.calc_G(use_masked=use_masked, product_only=product_only)
    conv_corr = d.convergence_corr()
    return covs, G, conv_corr, (d.R, d.T, d.L)

def covs_from_twopop(files, end=None, return_object=False,
                     *args, **kwargs):
    file_1, file_2 = files
    freqs_1 = sf.parse_slim_ragged_freqs(file_1)
    freqs_2 = sf.parse_slim_ragged_freqs(file_2)
    assert(freqs_1.freqs.shape == freqs_2.freqs.shape)
    times = np.arange(freqs_1.freqs.shape[0])
    assert(freqs_1.positions == freqs_2.positions)
    fqs_1 = freqs_1.freqs
    fqs_2 = freqs_2.freqs
    if end is not None:
        fqs_1 = fqs_1[np.arange(end), :]
        fqs_2 = fqs_2[np.arange(end), :]
        times = times[np.arange(end)]
    freqs = np.concatenate((fqs_1, fqs_2), axis=0)
    pops = [0] * fqs_1.shape[0] + [1] * fqs_2.shape[0]
    if return_object:
        # we include the loci here
        return freqs_to_covmat(freqs, times, pops,
                               return_object=True,
                               loci=freqs_1.positions)
    return ((freqs_1.params, freqs_2.params),
            *freqs_to_covmat(freqs, times, pops, *args, **kwargs))


def freqs_to_cov(frqs, burnin = 10000, gens=150,
                 sampled_gens=None, with_total_var=False,
                 fixed_to_nan=False, verbose=False):
    """
    More custamizable function for taking neutral frequencies and
    converting them to TemporalFreqs objects, calculating covariances,
    Gs.
    """
    use_masked = fixed_to_nan
    if sampled_gens is not None:
        idx = np.array([i for i, time in enumerate(frqs.samples) if
                        (burnin <= time <= burnin + gens) and
                        (time in sampled_gens)])
    else:
        idx = np.array([i for i, time in enumerate(frqs.samples) if
                        burnin <= time <= burnin + gens])
    samples = [(0, time) for time in idx]
    freqmat = frqs.freqs[idx, :]
    if fixed_to_nan:
        freqmat[np.logical_or(freqmat == 0., freqmat == 1.)] = np.nan
    d = TemporalFreqs(freqmat, samples)
    covs = d.calc_cov(use_masked=use_masked)
    if verbose:
        print(f"done calculating covs...")
    G = d.calc_G(use_masked=use_masked)
    if verbose:
        print(f"done calculating G...")
    if with_total_var:
        var = [d.calc_var(t=t) for t in range(d.T)]
        return frqs.params, covs, G, (d.R, d.T), var
    return frqs.params, covs, G, (d.R, d.T)

def covs_from_file(file, verbose=False, with_total_var=False, *args, **kwargs):
    covs = freqs_to_cov(sf.parse_slim_ragged_freqs(file),
                        with_total_var=with_total_var,
                        *args, **kwargs)
    if verbose:
        print(f"loading file {file} done.")
    return covs

def extract_runs(params, runs):
    kept_runs = defaultdict(list)
    for run_params, runs in runs.items():
        rp = dict(run_params)
        # include if this run's parameters include the values
        # in params
        if all([rp[k] in v for k, v in params.items()]):
            kept_runs[run_params].extend(runs)
    return kept_runs


def load_pickled(dir, exclude_md = ('seed', 'nrep', 'subpop'),
                 converters=None, add_in_Va=False):
    """
    Pickled results:
        metadata, covs, G, dimensions

    Removes exclude_md from params and builts a defaultdict of runs.

    Note: add_in_Va is a hack, as I forgot to add in the Va entry into
    metadata for some simulations (too computionally burdensome to rerun).
    """
    if converters is None:
        converters = {}
    results = defaultdict(list)
    for file in sorted(os.listdir(dir)):
        with open(dir + file, 'rb') as f:
            res = pickle.load(f)
            if isinstance(res[0], tuple):
                # validate the metadata for these files is matching
                p1, p2 = [tuple([(k, v) for k, v in x.items() if k not
                                    in exclude_md]) for x in res[0]]
                assert(p1 == p2)
                p = [*p1]
            else:
                p = [(k, v) for k, v in res[0].items() if
                            k not in exclude_md]
            if add_in_Va:
                p.append(("Va", re.match(r'.*_(.*)Va.*', file).groups()[0]))
                p.append(("L", re.match(r'.*_(.*)L.*', file).groups()[0]))
            p_converted = [(k, converters.get(k, str)(v)) for k, v in sorted(p)]
            results[tuple(p_converted)].append(res[1:])
    return results

def load_stats(dir, pattern=None, exclude_md=('seed', 'nrep'),
               converters=None, add_in_Va=False):
    """
    Note: add_in_Va is a hack, as I forgot to add in the Va entry into
    metadata for some simulations (too computionally burdensome to rerun).
    """
    if not dir.endswith('/'):
        dir += '/'
    if converters is None:
        converters = {}
    results = defaultdict(list)
    files = glob.glob(f"{dir}/*_stats.tsv")
    if pattern is not None:
        ddir = dir.replace('.', '\.')
        # print(f"{ddir}" + pattern)
        pattern_re = re.compile(f"{ddir}" + pattern)
    for file in sorted(files):
        if pattern is not None:
            if pattern_re.search(file) is None:
                continue
        res = sf.parse_slim_stats(file)
        p = list(res.params.items())
        if add_in_Va:
            p.append(("Va", re.match(r'.*_(.*)Va.*', file).groups()[0]))
            p.append(("L", re.match(r'.*_(.*)L.*', file).groups()[0]))
        p = [(k, converters.get(k, str)(v)) for k, v in sorted(p) if k not in exclude_md]
        results[tuple(p)].append(res.stats)
    return results

def params_vary_over(params):
    uniq = defaultdict(set)
    for param in params:
        if isinstance(param, tuple):
            param = dict(param)
        for key, val in param.items():
            uniq[key].add(val)
    return uniq


def col_palette(param, cmap, reverse=False, max=1):
    assert(len(param) == len(set(param)))
    n = len(param)
    x = np.linspace(0, max, n) if not reverse else np.linspace(max, 0, n)
    cols = cmap(x)
    return {k: cols[i, :] for i, k in enumerate(sorted(param))}

def average_runs(results, has_corr=True, has_var=False):
    """Average all replicate covariances and Gs

    has_corr: backwards compatability for runs without convergence correlation
    has_var: more hacks, this time for block covariances where we need to track
              total variance.

    """
    #Gs, covs = {}, {}
    out = {}
    for params, runs in results.items():
        Gs = np.stack(map(itemgetter(1), runs))
        covs = np.stack(map(itemgetter(0), runs))
        if has_corr:
            conv_corrs = np.stack(map(itemgetter(2), runs))
            dims = list(map(itemgetter(3), runs))
            assert(len(set([(r, t) for r, t, _ in dims])) == 1)
            out[params] = covs, Gs, conv_corrs, dims[0]
        elif has_var:
            vars = np.stack(map(itemgetter(3), runs))
            dims = list(map(itemgetter(2), runs))
            assert(len(set([(r, t) for r, t in dims])) == 1)
            out[params] = covs, Gs, vars, dims[0]
        else:
            dims = list(map(itemgetter(2), runs))
            assert(len(set([(r, t) for r, t in dims])) == 1)
            out[params] = covs, Gs, dims[0]
    return out


def CI_polygon(x, lower, upper, smooth=True, frac=1./10, **kwargs):
    if smooth:
        lowess = sm.nonparametric.lowess
        lower = lowess(lower, x, frac=frac)[:, 1]
        upper = lowess(upper, x, frac=frac)[:, 1]
    #verts = [(np.min(x), 0), *zip(lower, upper), (np.max(x), 0)]
    verts = list(zip(x, lower)) + list(zip(reversed(x), reversed(upper)))
    poly = Polygon(verts, **kwargs)
    return poly
