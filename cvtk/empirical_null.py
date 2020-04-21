from itertools import groupby
import numpy as np
from tqdm import tnrange

from cvtk.cov import stack_temporal_covariances, calc_deltas
from cvtk.cov import cov_by_group, temporal_replicate_cov
from cvtk.utils import sliceify


def slice_loci(seqids, loci_indices=None, exclude_seqids=None):
    """
    seqids is a vector of the Sequence ID for each locus, in order of the loci
    of the array to be sliced.
    """
    loci_slice = slice(None)
    if exclude_seqids is None:
        exclude_seqids = set()
    if loci_indices is None:
        idx = [i for i, seqid in enumerate(seqids) if seqid not in exclude_seqids]
    else:
        loci_indices = set(loci_indices)
        idx = [i for i, seqid in enumerate(seqids) if seqid not in exclude_seqids and
                i in loci_indices]
    #loci_slice = np.array(idx)
    return sliceify(idx)


def sign_permute_delta_matrix(deltas, blocks=None):
    """
    Permute and flip the sign of the timepoints in the
    R x T x L deltas matrix.
    """
    assert(deltas.ndim == 3)
    R, T, L = deltas.shape
    resampled_deltas = np.array(deltas)  # copy
    if blocks is None:
        flips = np.tile(np.random.choice((-1, 1), size=(T, 1)), L)
    else:
        nblocks = len(set(blocks))
        # TODO make a flip per block
        #assert(sorted(blocks) == blocks)
        block_reps = [len(list(x)) for group, x in groupby(blocks)]
        flips = np.repeat(np.random.choice((-1, 1), size=(nblocks, T)), block_reps, axis=0)
    resampled_deltas *= flips.T
    return resampled_deltas


def reshape_empirical_null(empnull, R, T):
    empcovs = list()
    for strap in empnull:
        tile_empcovs = list()
        for tile_covmat in strap:
            temp_covmats = stack_temporal_covariances(tile_covmat, R, T)
            tile_empcovs.append(temp_covmats)
        empcovs.append(np.stack(tile_empcovs))
    return np.stack(empcovs)


def calc_covs_empirical_null(freqs, tile_indices, tile_seqids,
                             tile_ids, gintervals,
                             B=100,
                             depths=None, diploids=None,
                             by_tile=False,
                             use_masked=False,
                             exclude_seqids=None,
                             sign_permute_blocks='tile', bias_correction=False,
                             progress_bar=True):
    """
    Params:
           - sign_permute_blockse: either 'seqid' or 'tile', whether to sign-flip by
              whole chromosomes or by tile (less conservative).
           - by_tile: whether to run the empirical null on the tile covariance matrices
               (True) or the genome-wide covariance matrix.
    """
    permuted_covs = list()
    # If we are doing things by tiles, we need to only include the loci in all
    # tiles, e.g. if some tiles drop certain loci at the ends.
    if sign_permute_blocks == 'tile':
        tile_loci = [locus for tile in tile_indices for locus in tile]
        loci_slice = slice_loci(gintervals.seqid, tile_loci, exclude_seqids)
        # need to change the seqid vector, which is for every locus, to only include
        # the loci in the tiles
        seqids = tile_seqids
    elif sign_permute_blocks == 'seqid':
        # just handle slicing out certin seqids
        loci_slice = slice_loci(gintervals.seqid, exclude_seqids=exclude_seqids)
        seqids = gintervals.seqid
    else:
        raise ValueError("sign_permute_blocks must be 'tile' or 'seqid'")

    # slice all the appropriate data structures
    deltas = calc_deltas(freqs)
    R, T, L = deltas.shape
    sliced_freqs = freqs[..., loci_slice]
    sliced_deltas = deltas[:, :, loci_slice]
    # diploids does not need to be sliced, since we can broadcast the last dimension
    sliced_depths, sliced_diploids = None, diploids
    if depths is not None:
        sliced_depths = depths[..., loci_slice]

    if progress_bar:
        B_range = tnrange(int(B))
    else:
        B_range = range(int(B))

    # main delta permutation procedure
    all_covs = list()
    for b in B_range:
        # this permutes timepoints, and randomly flips the
        # sign for entire blocks loc loci, keeping the replicates
        # and loci unchanged. By specifying the chromosomes
        # as blocks, we allow differential flipping across chromosomes
        if sign_permute_blocks == 'tile':
            permuted_deltas = sign_permute_delta_matrix(sliced_deltas, tile_ids)
        # NOTE: I think we can do without the above, not artificially breaking up the
        # dependencies in the data.
        elif sign_permute_blocks == 'seqid':
            permuted_deltas = sign_permute_delta_matrix(sliced_deltas, seqids)
        else:
            raise ValueError("sign_permute_blocks must be either 'tile' or 'seqid'")

        if by_tile:
            covs = cov_by_group(tile_indices, sliced_freqs,
                                 depths=sliced_depths,
                                 diploids=sliced_diploids,
                                 bias_correction=bias_correction,
                                 deltas=permuted_deltas, use_masked=use_masked)
        else:
            covs = temporal_replicate_cov(sliced_freqs, depths=sliced_depths,
                                          diploids=sliced_diploids,
                                          bias_correction=bias_correction,
                                          deltas=permuted_deltas,
                                          use_masked=use_masked)

        all_covs.append(covs)
    if by_tile:
        return reshape_empirical_null(all_covs, R, T)
    return all_covs

