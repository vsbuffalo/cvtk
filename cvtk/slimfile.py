## slim.py -- parse slim results in a tabular format
import numpy as np
import pandas as pd
import glob
import os
import re
import sys
import random
from collections import namedtuple, OrderedDict, defaultdict
from itertools import product
import pandas as pd
from scipy.sparse import coo_matrix

SlimFreqs = namedtuple('SlimFreqs', ('params', 'positions', 'samples',
                                     'freqs', 'times'))
SlimStats = namedtuple('SlimStats', ('params', 'stats'))

def split_keyval(keyval):
    key, val = keyval.split('=')
    if key.isalpha():
        return key, val
    return key, float(val)

def parse_params(param_str):
    "Parse the parameter string."
    return dict([split_keyval(keyval) for keyval in param_str.split(';')])


def parse_slim_stats(filename, delim='\t', verbose=False):
    """
    Parse a tabular representation of population statistics.

    The return value is a (params, stats) tuple.
    """
    with open(filename) as fp:
        line = next(fp)  # graph first line
        assert(line.startswith('#'))
        params = parse_params(line[1:])
        # reset the read position
        fp.seek(0)
        statsdf = pd.read_csv(fp, comment='#', header=0, delimiter=delim)
    return SlimStats(params, statsdf)


def parse_slim_ragged_freqs(filename, delim='\t'):
    """
    Parse a ragged array of
      'gen', 'id;pos;freq' x number of polymorphic mutations

    Assumes generations are in order. Uses sparse matrices to efficiently
    load these data into a matrix.
    """
    muts = dict()
    row, col, data = [], [], []
    first_gen = None
    gens = []
    # mock genomic intervals
    positions = []
    # we keep our own internal IDs here
    with open(filename) as fp:
        line = next(fp)
        params = parse_params(line[1:])
        for line in fp:
            fields = line.strip().split(delim)
            gen = int(fields[0])
            gens.append(gen)
            if first_gen is None:
                first_gen = gen
            for mut in fields[1:]:
                mfs = mut.split(';')
                mid, pos, freq = int(mfs[0]), int(mfs[1]), float(mfs[2])
                muts[mid] = pos
                positions.append(pos)
                row.append(gen-first_gen)
                col.append(mid)
                data.append(freq)
    # remap the timepoints
    timepoints = {t: i for i, t in enumerate(set(row))}
    row = [timepoints[t] for t in row]
    # now map their IDs to our IDs
    key_map = dict((k, i) for i, k in enumerate(muts.keys()))
    new_col = [key_map[mid] for mid in col]
    assert(len(new_col) == len(row) == len(data))
    loci = OrderedDict((muts[mid], i) for i, mid in enumerate(col))
    # Even though SLIM doesn't really have concenpt of multiple chromosomes,
    # we use the defaultdict(OrderedDict) approach as in handling SyncFiles.
    # The chromosome name is None
    loci_dict = defaultdict(OrderedDict)
    loci_dict[None] = loci
    return SlimFreqs(params, positions, gens,
                     coo_matrix((data, (row, new_col)),
                                shape=(len(gens), len(new_col))).toarray(),
                     timepoints)


def parse_slim_freqs(filename, delim='\t', min_prop_samples=0,
                     missing_val=np.nan, verbose=False):
    """
    Parse a tabular representation of output frequencies.

    In my SLiM results, I preallocate a ngens x nloci matrix, and fill it with
    results as generations complete. Empty values are assigned a value of -1,
    which is replaced with `missing_val` (default: np.nan).

    The first row is a parameter string beginning with #, with 'key=val'
    parameter values.

    The return value is a (params, generations, frequency matrix) tuple.
    """
    with open(filename) as fp:
        next(fp)  # skip parameters
        header = next(fp).strip().split(delim)
        loci = np.array(header[1:], dtype='u4')  # grab loci
        # get the number of columns
        width = len(fp.readline().strip().split(delim))
        # reset the read position
        fp.seek(0)
        # parse the params
        line = next(fp).strip()
        if not line.startswith('#'):
            msg = ("error: SLiM results file does not begin with #-prefixed "
                   "string containing parameters.")
            raise ValueError(msg)
        params = parse_params(line[1:])
        fp.seek(0)
        next(fp); next(fp) # skip parameters, skip header
        # now, read the entire matrix
        #dtypes = ', '.join(['i4'] + ['f4'] * (width - 1))
        dtypes = 'float64'
        mat = np.loadtxt(fp, delimiter=delim, dtype=dtypes)
    # extract out the generation (samples here) and convert types
    samples = mat[:,0].astype('i4')
    # drop generation column, and convert -1 to nan
    mat = np.delete(mat, 0, 1)
    # convert -1s for non-poymorphic site to 0s
    mat[mat < 0] = missing_val
    # remove loci that are never polymorphic
    min_nsamples = int(min_prop_samples*mat.shape[0])
    keep_cols = np.logical_not(np.isnan(mat)).sum(0) > min_nsamples
    if verbose:
        msg = (f"pruning matrix from {mat.shape[1]} to "
               f"{keep_cols.sum()} columns (threshold: >{min_nsamples} samples)")
        print(msg)
    mat = mat[:, keep_cols]
    loci = loci[keep_cols]
    return SlimFreqs(params, loci, samples, mat)

def parse_metadata(file, converters=None):
    metadata = []
    with open(file, 'r') as f:
        for line in f:
            if line.startswith("#"):
                metadata.append(line[1:].strip())
            else:
                break
    mds = "".join(metadata).split(";")
    md_dict = dict()
    for md in mds:
        key, val = md.split("=")
        assert(key not in md_dict)
        if converters is not None:
            md_dict[key] = converters[key](val)
        else:
            md_dict[key] = val
    return md_dict


def get_suffix(file, suffixes):
    matches = 0
    for suffix in suffixes:
        if suffix in file:
            match = suffix
            matches += 1
    assert(matches == 1)
    return suffixes[match]


class SimResults(object):
    def __init__(self, dir, suffixes, pattern=None,
                 converters=None, replicate=True):
        self.dir = dir
        self.files = sorted(os.listdir(dir))
        if pattern is not None:
            pattern_re = re.compile(pattern)
            self.files = [f for f in self.files if pattern_re.search(f)
                          is not None]
        self.paths = [os.path.join(dir, f) for f in self.files]
        # suffixes is a dict of file suffix -> name
        self.suffixes = suffixes
        self.replicate = replicate
        self._converters = converters
        self.raw_results = self._group_runs()
        self.results = pd.DataFrame(self._invert_runs())

    def _group_runs(self):
        "Group runs by removing their file-specific suffixes"
        runs = dict()
        for file, path in zip(self.files, self.paths):
            # classify each file by suffix key
            key = None
            for suffix in self.suffixes:
                # find at least one key in this file
                if suffix in file:
                    key = file.replace(suffix, "")
            assert(key is not None)
            if self.replicate:
                # remove the replicate number at the end
                # to group by unique parameters
                okey = key
                key = re.sub("_(\d+)_?$", "", key)

            if key not in runs:
                # initialize a new element
                runs[key] = {s: [] for s in self.suffixes.values()}
                runs[key]['metadata'] = None


            # get the suffix of this file type
            suffix = get_suffix(file, self.suffixes)
            # if the we are to parse out the replicate number...
            runs[key][suffix].append(path)

            # metadata
            md = parse_metadata(path, converters=self._converters)
            if runs[key]['metadata'] is not None:
                # we verify the metadata is the same -- a sanity check
                # if there's a seed metdata parameter, remove that
                # for comparision (and subpop and nrep!)
                remove = ('seed', 'subpop', 'nrep')
                md_alt = runs[key]['metadata']
                md_no_seed = {k: v for k, v in md.items() if k not in remove}
                md_alt_no_seed = {k: v for k, v in md_alt.items() if k  not in remove}
                assert(md_alt_no_seed == md_no_seed)
            else:
                runs[key]['metadata'] = md

        return runs

    def _invert_runs(self):
        "Take raw results and shape into something Pandas likes"
        cols = defaultdict(list)
        for key, result in self.raw_results.items():
            cols['key'].append(key)
            for md, val in result['metadata'].items():
                cols[md].append(val)
            cols['metadata'].append(result['metadata'])
            for k, val in result.items():
                if k == 'metadata':
                    continue
                cols[k + '_file'].append(val)
        return cols

def output_filename(input_filename, suffix, ext='.tsv'):
    _, inext = os.path.splitext(input_filename)
    return input_filename.replace(inext, f'-{suffix}{ext}')

if __name__ == '__main__':
    import sys
    import tempdata

    file = sys.argv[2]
    cmd = sys.argv[1]
    simdata = parse_slim_tsv(file, min_prop_samples=0.)
    d = tempdata.TemporalFreqs(simdata.freqs, simdata.samples, simdata.loci)

    if cmd == 'cov':
        print("calculating covariances...")
        d.calc_covs()
        print("writing covariances...")
        d.write_covs(output_filename(file, 'cov'), long=True)
    elif cmd == 'freq':
        print("writing frequencies...")
        d.write_freqs(output_filename(file, 'freq'))
    else:
        print("usage: [cov,freq] input_file.txt")
        sys.exit(1)
