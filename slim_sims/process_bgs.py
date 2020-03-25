# necessary to load cvtk code:
import os
import sys
import glob
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import pickle
import random
from multiprocessing import Pool
from collections import defaultdict
import pandas as pd
import numpy as np

from cvtk.cvtk import TemporalFreqs, TiledTemporalFreqs
from cvtk.cov import stack_temporal_covariances
import cvtk.slimfile as sf

def freqs_to_cov(frqs, burnin = 10000, gens=150, with_G=True,
                  fixed_to_nan=False, verbose=True):
    use_masked = fixed_to_nan
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
    if with_G:
        G = d.calc_G(use_masked=use_masked)
        if verbose:
            print(f"done calculating G...")
        return covs, G
    return covs

def covs_from_file(file, verbose=True, *args, **kwargs):
    freqs = freqs_to_cov(sf.parse_slim_ragged_freqs(file), *args, **kwargs)
    if verbose:
        print(f"loading file {file} done.")
    return freqs

def downsample_vector(x, subsample):
    return random.sample(x, subsample)

## load the files
conv = {"seed": int, "s": float, "rbp": float, "region_length": int, "nmu": float,
        "smu": float, "U":float, "N": int}

res = sf.SimResults('../data/sims/bgs/',
                    pattern = "bgs_1000N_1e-08rbp_.*s_1e-08nmu_.*U.*",
                    suffixes={"_neutfreqs.tsv": "neutral_freqs",
                              "_stats.tsv": "stats"},
                    converters=conv)
df = res.results

# calculate the covariances / G
nsamples = 30
print(f"entries to process: {nsamples * df.shape[0]}")
BGS_COVS = "../data/sims_intermediate/bgs_covs.pkl"
pool = Pool(processes = 10)

if not os.path.exists(BGS_COVS):
    all_covs = defaultdict(list)
    n = df.shape[0]
    keys = df.key.values.tolist()
    for i, reps in enumerate(df.neutral_freqs_file.values.tolist()):
        key = keys[i]
        print(key)
        print(f"starting processing for simulation parameter set {i}/{n}...\t", end="")
        rep_covs = pool.map(covs_from_file, downsample_vector(reps, nsamples))
        print("done.")
        all_covs[key].extend(rep_covs)
    with open(BGS_COVS, 'wb') as f:
        pickle.dump(all_covs, f)
else:
    with open(BGS_COVS, 'rb') as f:
        all_covs = pickle.load(f)

