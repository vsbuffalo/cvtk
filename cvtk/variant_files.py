"""
Some interfaces to common variant file formats.

Warning: these are a bit inconsistent, and some highly
specialized for the data sets I analyze, e.g. the AD_DP_VCFFile()
class which is tailored for the Bergland et al (2014)'s dataset.
In the future I'll refactor this to make it more consistent.

"""
import sys
import time
import gzip
import operator as op
from collections import defaultdict, namedtuple, OrderedDict
import h5py
import numpy as np
import pandas as pd
import allel

from cvtk.gintervals import GenomicIntervals


Locus = namedtuple('Locus', ('chr', 'pos', 'ref', 'major', 'minor'))

SYNC_ALLELES = ('A', 'T', 'C', 'G', 'N', 'd')
SYNC_ALLELES_DTYPE = [(allele, 'u4') for allele in SYNC_ALLELES]

def convert_np_object_to_string(x):
    strlen = max(set(map(len, x)))
    return x.astype(f"S{strlen}")


def split_sample_allele_counts(samp_cols):
    "Split the sync file's string into counts matrix"
    mat = np.array([list(map(int, samp.split(':'))) for samp in samp_cols],
                   dtype='u4')
    return mat

def syncfile_filter_diallelic(mat):
    """Subset a matrix of allele counts, keeping only the two most frequent
    alleles. Returns a tuple, ((major allele, minor allele), counts matrix).
    """
    counts = mat.sum(0)
    seg_indices = np.flip(counts.argsort())
    return ((SYNC_ALLELES[seg_indices[0]], SYNC_ALLELES[seg_indices[1]]),
            mat[:, seg_indices[0:2]])

def get_sizes(iter):
    return sum(sys.getsizeof(x) for x in iter)


class VariantFile(object):
    def __init__(self, filename, verbose=True, seqlens=None, max_seqname_len=5):
        "Open and parse a sync file, keeping diallelic loci."
        self.filename = filename
        self.mat = list()
        self._size = 0
        self.nsamples = 0
        self.samples = None
        self.nloci = 0
        self.freqs = None
        # for HDF5, we store all these as auxillary columns
        self.mat = []
        self.chrom_names = None
        self.chroms, self.positions = [], []
        self.refs, self.majors, self.minors = [], [], []
        self.verbose = verbose

    @property
    def N(self):
        return self.mat.sum(axis=2)

    def build_gintervals(self, include_alleles=False):
        g = GenomicIntervals()
        for chrom, pos in zip(self.chroms.astype(str), self.positions):
            g.append(chrom, pos)
        if include_alleles:
            g.set_data(pd.DataFrame(dict(ref=self.refs.astype(str),
                                     major=self.majors.astype(str),
                                     minor=self.minors.astype(str))))
        return g

    def remove_fixed(self):
        if self.freqs is None:
            raise ValueError("Calculate frequencies first with VariantFile.calc_freqs()")
        freqs = self.freqs
        to_keep = np.logical_not(np.apply_along_axis(all, 0, (freqs == 0) | (freqs == 1)))
        to_keep_indices = np.where(to_keep)[0]
        self.chroms = self.chroms[to_keep_indices]
        self.positions = self.positions[to_keep_indices]
        self.mat = self.mat[:, to_keep_indices, :]
        self.freqs = self.freqs[:, to_keep_indices]
        if len(self.refs):
            self.refs = self.refs[to_keep_indices]
        if len(self.majors):
            self.majors = self.majors[to_keep_indices]
        if len(self.minors):
            self.minors = self.minors[to_keep_indices]

    def __repr__(self):
        return f"VariantFile with {self.nloci} loci and {self.nsamples} samples."

    def prune_samples(self, keep_samples):
        # covernt to string
        samples = [x.decode() for x in self.samples]
        # otherwise, will iterate over characters
        assert(not isinstance(keep_samples, str))
        keep_samples = set(keep_samples)
        idx = np.array([i for i, sample in enumerate(samples) if sample in keep_samples])
        self.samples = [sample for sample in samples if sample in keep_samples]
        self.geno_mat = self.geno_mat[:, idx, :]
        self.mat = None

    def create_allele_counts(self, subpops):
        """
        Create an allele counts matrix.
        """
        try:
            out = allel.GenotypeArray(self.geno_mat).count_alleles_subpops(subpops)
        except Exception as er:
            msg = ("scikit-allel's GenotypeArray.count_alleles_subpops() failed, "
                    "likely because subpops was not the correct data structure. "
                    "It should be a dictionary of subpopulation keys and "
                    "a list of integer index values.")
            print(msg)
            raise er
        key_order = sorted(out.keys())
        self.mat = np.stack([out[k] for k in key_order])
        self.subpops = key_order


    @property
    def size(self):
        return f"{self._size / 1e6:,.2f} Mb"

    @classmethod
    def load_hdf5(cls, filename, calc_freqs=True, verbose=True):
        t0 = time.time()
        obj = VariantFile(None)
        with h5py.File(filename, 'r') as hf:
            obj.mat = hf["counts_matrix"][()]
            obj.nloci = obj.mat.shape[0]
            obj.nsamples = obj.mat.shape[1]
            loci_grp = hf["loci"]
            chrom_names = loci_grp["chrom_names"]
            chrom_map = dict((i, k) for i, k in enumerate(chrom_names))
            obj.chroms = np.string_([chrom_map[c] for c in loci_grp['chroms'][()]])
            obj.positions = loci_grp["positions"][()]
            obj.refs = loci_grp["refs"][()]
            obj.majors = loci_grp["majors"][()]
            obj.minors = loci_grp["minors"][()]
            if 'samples' in loci_grp:
                obj.samples  = loci_grp["samples"][()]
            else: 
                obj.samples = None
            obj.filename = hf["counts_matrix"].attrs["filename"].decode()
            obj._size = hf["counts_matrix"].attrs["size"]
        t1 = time.time()
        if verbose:
            print(f"total time to load HDF5 file: {(t1-t0)/60} mins.")
        if calc_freqs:
            obj.calc_freqs()
        return obj


    def dump_hdf5(self, filename, verbose=True):
        t0 = time.time()
        with h5py.File(filename, 'w') as hf:
            # genotype data
            dset = hf.create_dataset("counts_matrix", self.mat.shape,
                                     data=self.mat, dtype='u4')
            # serialize the loci data
            loci_grp = hf.create_group("loci")
            totlen = (self.nloci, )
            # create the chromosome enum
            chrom_names = np.string_(sorted(self.chrom_names))
            cnS = max(map(len, chrom_names))
            loci_grp.create_dataset('chrom_names', (len(chrom_names), ),
                                    data=chrom_names,
                                    dtype=f"S{cnS}")
            # store chroms as enum
            chrom_map = dict((k, i) for i, k in enumerate(chrom_names))
            chrom_dt = h5py.special_dtype(enum=('i', chrom_map))
            chrs = np.array([chrom_map[c] for c in self.chroms], dtype='i')
            # store additional data about loci
            loci_grp.create_dataset('chroms', totlen, data=chrs, dtype=chrom_dt)
            loci_grp.create_dataset('positions', totlen,
                                    data=self.positions, dtype='u4')
            loci_grp.create_dataset('refs', totlen, data=self.refs,
                                    dtype='S1')
            loci_grp.create_dataset('majors', totlen, data=self.majors,
                                    dtype='S1')
            loci_grp.create_dataset('minors', totlen, data=self.minors,
                                    dtype='S1')
            if self.samples is not None:
                sample_strlen = max(map(len, set(self.samples)))
                loci_grp.create_dataset('samples', len(self.samples), 
                                        data=self.samples,
                                        dtype=f"S{sample_strlen}")
            # metadata
            dset.attrs['filename'] = np.string_(self.filename)
            dset.attrs['size'] = np.array(self._size, dtype='u4')
        t1 = time.time()
        if verbose:
            print(f"total time to dump HDF5 file: {(t1-t0)/60} mins.")

    def calc_freqs(self):
        "Calculate the frequencies from the count data."
        # self.mat is list nloci long, each a matrix of ntime x 2 long
        # with the two columns containing the counts of major/minor alleles
        denom = self.mat.sum(axis=2)
        self.freqs = self.mat[:,:,0] / denom
        return self.freqs

class SyncFile(VariantFile):
    def __init__(self, filename, verbose=True, seqlens=None, max_seqname_len=5):
        super().__init__(filename, verbose=True, seqlens=None, max_seqname_len=5)
        if filename is not None:
            self._load_sync()

    def _load_sync(self):
        t0 = time.time()
        first = True
        row_i = 0  # for mapping between the chrom/loci to matrix
        with gzip.open(self.filename, 'r') as sf:
            if self.verbose:
                print(f"reading file '{self.filename}'...")
            for lineno, line in enumerate(sf):
                row = line.decode("utf-8").strip().split('\t')
                # Main columns.
                chrom, pos, ref = row[0], int(row[1]), row[2]
                # Sample counts.
                counts_mat = split_sample_allele_counts(row[3:])

                # Prune to only include two polymorphic alleles.
                # If it looks like the locus seg. alleles > 2, skip
                alleles, counts_mat = syncfile_filter_diallelic(counts_mat)

                # store non-genotype data
                self.chroms.append(chrom)
                self.positions.append(pos)
                self.refs.append(ref)
                self.majors.append(alleles[0])
                self.minors.append(alleles[1])
                if first:
                    self.nsamples = counts_mat.shape[0]
                    first = False
                else:
                    if self.nsamples != counts_mat.shape[0]:
                        msg = (f"malformed .sync file: number of samples " +
                               "varies across rows (line: {lineno})")
                        raise ValueError(msg)

                # Append to matrix, and update mapping of chrom/loci to matrix.
                self.mat.append(counts_mat)

                # Update the size attribute.
                self._size += counts_mat.nbytes
                row_i += 1
                if lineno % 1e3 == 0:
                    print(f"  {lineno} lines processed", end='\r')
                self.nloci += 1

        self.mat = np.stack(self.mat)
        self.chrom_names = set(self.chroms)
        self.chroms = np.string_(self.chroms)
        self.positions = np.array(self.positions)
        self.refs = np.string_(self.refs)
        self.majors = np.string_(self.majors)
        self.minors = np.string_(self.minors)
        if self.verbose:
            print(f"file '{self.filename}' loaded.")
            t1 = time.time()
            print(f"total time to load Sync file: {(t1-t0)/60} mins.")
        # calculate the frequencies from the counts
        self.calc_freqs()


class VCFFile(VariantFile):
    def __init__(self, filename, verbose=True, seqlens=None):
        super().__init__(filename, verbose=verbose, seqlens=seqlens)
        self._load_vcf()

    def _load_vcf(self):
        t0 = time.time()
        if self.verbose:
            print(f"reading file '{self.filename}'...")
        vcf = allel.read_vcf(self.filename)
        self.chroms =  convert_np_object_to_string(vcf['variants/CHROM'])
        self.chrom_names =  set(self.chroms)
        self.refs = convert_np_object_to_string(vcf['variants/REF'])
        self.alts = convert_np_object_to_string(vcf['variants/ALT'])
        self.samples = convert_np_object_to_string(vcf['samples'])
        self.positions = vcf['variants/POS']
        self.geno_mat = allel.GenotypeArray(vcf['calldata/GT'])
        self.mat = None
        if self.verbose:
            print(f"file '{self.filename}' loaded.")
            t1 = time.time()
            print(f"total time to load VCF file: {(t1-t0)/60} mins.")
        # calculate the frequencies from the counts

    def count_alleles_subpops(self, sample_groups):
        """
        A wrapper around scikit-allel's GenotypeArray.count_alleles_subpops() method, 
        which takes a dictionary of subpop name keys and values that are the indices
        in that subpopulation.
        """
        counts_mat = self.geno_mat.count_alleles_subpops(sample_groups)
        self.mat = np.stack(counts_mat.values())
        self.subpops = sample_groups.keys()
        return counts_mat

class AD_DP_VCFFile(VariantFile):
    """
    This is an odd VCF file from Bergland et al 2014, where only sample AD:DP is stored,
    (alt allele depth and depth).
    """
    def __init__(self, filename, verbose=True, seqlens=None):
        super().__init__(filename, verbose=verbose, seqlens=seqlens)
        self._load_vcf()

    def _load_vcf(self):
        t0 = time.time()
        if self.verbose:
            print(f"reading file '{self.filename}'...")
        self.chroms =  []
        self.refs = []
        self.alts = []
        self.samples = None
        self.positions = []
        ad = []
        dp = []

        with gzip.open(self.filename, 'rb') as f:
            metadata = []
            for line in f:
                line = line.decode().strip()
                if line.startswith('##'):
                    metadata.append(line.strip('##'))
                    continue
                if line.startswith('#CHR'):
                    fields = line.strip('#').split('\t')
                    self.samples = fields[9:]
                    continue
                # proceed with data
                fields = line.strip('#').split('\t')
                chr, pos, id, ref, alt, qual, filter, info, format = fields[:9]
                self.chroms.append(chr)
                self.positions.append(int(pos))
                self.refs.append(ref)
                self.alts.append(alt)
                assert(format == "AD:DP")
                sample_ad, sample_dp = zip(*[map(int, x.split(':')) for x in fields[9:]])
                ad.append(sample_ad)
                dp.append(sample_dp)

        ad = np.array(ad) 
        dp = np.array(dp) 
        self.mat = np.stack([ad, dp-ad], axis=2)
        self.chrom_names = set(self.chroms)
        self.chroms = np.string_(self.chroms)
        self.positions = np.array(self.positions)
        self.refs = np.string_(self.refs)
        self.majors = np.string_(self.majors)
        self.minors = np.string_(self.minors)
 
        if self.verbose:
            print(f"file '{self.filename}' loaded.")
            t1 = time.time()
            print(f"total time to load VCF file: {(t1-t0)/60} mins.")
        # calculate the frequencies from the counts
        self.calc_freqs()
 

if __name__ == "__main__":
    import sys
    import pickle

    if __name__ == "__main__":
        if len(sys.argv) < 2:
            msg = ("to parse and serialize input.sync.gz, run: "
                   "python syncfile.py input.sync.gz")
            exit()
        filename = sys.argv[1]
        d = SyncFile(filename)
        outfilename = filename.replace('.sync.gz', '.pkl')
        with open(outfilename, 'wb') as pf:
            pickle.dump(d, pf)

