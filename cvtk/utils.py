from collections import Counter
import numpy as np

def reshape_matrix(matrix, R):
    """
    Take a S x L frequency or depth matrix (where S = R x T is the number of
    samples), and the number of replicates, and reshape the frequency matrix
    such that it's a R x T x L tensor.
    """
    # find the indices where the replicate switches -- as evident from checks
    # this relies on replicates and freqs in same order, sorted by replicates!
    return np.stack(np.vsplit(matrix, R))

def flatten_matrix(matrix, R, T, L):
    """
    Flatten a 3D array of shape R x T x L into an (R x T) x L matrix.  This
    relies on the specific way numpy reshapes arrays, specifically that
    *timepoints are nested within replicates*. This makes the process of
    calculating covariances easier, since the temporal covariance sub block
    matrices are all around the diagonal.

    Below is an example demonstrating this is the default np.reshape() behavior.
    If this were to ever change, this could would break.

    TODO: unit test.

    R = 2; T = 3; L = 5
    M = np.stack([np.array([[f"R={r},T={t},L={l}" for l in range(L)] for
                        t in range(T)]) for r in range(R)])

    M.shape
        (2, 3, 5)

    M  # this is the form of the tensor:
    array(
    [[['R=0,T=0,L=0', 'R=0,T=0,L=1', 'R=0,T=0,L=2', 'R=0,T=0,L=3', 'R=0,T=0,L=4'],
        ['R=0,T=1,L=0', 'R=0,T=1,L=1', 'R=0,T=1,L=2', 'R=0,T=1,L=3', 'R=0,T=1,L=4'],
        ['R=0,T=2,L=0', 'R=0,T=2,L=1', 'R=0,T=2,L=2', 'R=0,T=2,L=3', 'R=0,T=2,L=4']],

    [['R=1,T=0,L=0', 'R=1,T=0,L=1', 'R=1,T=0,L=2', 'R=1,T=0,L=3', 'R=1,T=0,L=4'],
        ['R=1,T=1,L=0', 'R=1,T=1,L=1', 'R=1,T=1,L=2', 'R=1,T=1,L=3', 'R=1,T=1,L=4'],
        ['R=1,T=2,L=0', 'R=1,T=2,L=1', 'R=1,T=2,L=2', 'R=1,T=2,L=3', 'R=1,T=2,L=4']]],
        dtype='<U11')

    M.reshape((R*T, L))  # this is the flattened version; note the structure
    # where the timepoints are grouped inside a replicate. This is the
    # appropriate structure for blocked covariance matrix.

    array(
    [['R=0,T=0,L=0', 'R=0,T=0,L=1', 'R=0,T=0,L=2', 'R=0,T=0,L=3', 'R=0,T=0,L=4'],
        ['R=0,T=1,L=0', 'R=0,T=1,L=1', 'R=0,T=1,L=2', 'R=0,T=1,L=3', 'R=0,T=1,L=4'],
        ['R=0,T=2,L=0', 'R=0,T=2,L=1', 'R=0,T=2,L=2', 'R=0,T=2,L=3', 'R=0,T=2,L=4'],

        ['R=1,T=0,L=0', 'R=1,T=0,L=1', 'R=1,T=0,L=2', 'R=1,T=0,L=3', 'R=1,T=0,L=4'],
        ['R=1,T=1,L=0', 'R=1,T=1,L=1', 'R=1,T=1,L=2', 'R=1,T=1,L=3', 'R=1,T=1,L=4'],
        ['R=1,T=2,L=0', 'R=1,T=2,L=1', 'R=1,T=2,L=2', 'R=1,T=2,L=3', 'R=1,T=2,L=4']],
        dtype='<U11')


    """
    assert(matrix.ndim == 3)
    return matrix.reshape((R * T, L))

def swap_alleles(freqs):
    # assumes ngens x nloci
    if freqs.ndim == 3:
        _, ngens, L = freqs.shape
    elif freqs.ndim == 2:
        ngens, L = freqs.shape
    else:
        raise ValueError("frequency matrix must be 2D or 3D")
    flips = np.broadcast_to(np.random.binomial(1, 0.5, L), (ngens, L))
    # flips is broadcast to all timepoints; we just return the loci's
    # flips for the same timepoint since all are identical.
    swap_alleles = flips[0]
    return np.abs(flips - freqs), flips


def sort_samples(samples):
    """
    Sort the samples, ordering and grouping by replicate, then timepoint.
    This returns a tuple, of the ordered samples and then an indexing array
    to rearrange the columns of the input frequency matrix.
    """
    sorted_i = sorted(range(len(samples)),
                      key=lambda i: (samples[i][0], samples[i][1]))
    return [samples[i] for i in sorted_i], sorted_i

def process_samples(freqs, samples):
    """
    Validate samples, ensuring they are:
    (1) the right type, a list of tuples, each tuple being
        (replicate, timepoint).
    (2) that samples are grouped by replicate and then timepoint.
    (3) the number of samples is equal to the number of rows of the
        frequency matrix.
    (4) the design is balanced (R is same for all timepoints)

    Then, if everything is proper, return a tuple of lists of each replicates
    and timepoints.

    """
    samples = list(samples)
    # check types
    try:
        assert(isinstance(samples, list))
        assert(all(isinstance(x, tuple) and len(x) == 2) for x in samples)
    except AssertionError:
        raise ValueError("samples must be a list of tuples, (replicate, timepoint).")

    # check sorting
    try:
        assert(sorted(samples, key=lambda x: (x[0], x[1])) == samples)
    except AssertionError:
        raise ValueError("samples must sorted by replicate, then timepoint, e.g. "
                        "(R1, T1), (R1, T2), (R1, T3), (R2, T1), (R2, T2), ...")

    # check dimension compatability
    if len(samples) != freqs.shape[0]:
        raise ValueError("len(samples) != number of rows in samples.")
    replicates, timepoints = zip(*samples)

    # check if design (ntimepoints/nreplicates is balanced)
    timepoints_counts = Counter(timepoints)
    replicates_counts = Counter(replicates)
    timepoints_is_balanced = len(set(timepoints_counts.values())) == 1
    replicates_is_balanced = len(set(replicates_counts.values())) == 1
    try:
        assert(timepoints_is_balanced)
    except AssertionError:
        msg = "timepoints are not balanced — equal number of replicates are needed"
        raise ValueError(msg)
    try:
        assert(timepoints_is_balanced)
    except AssertionError:
        msg = "replicates are not balanced — equal number of timepoints needed"
        raise ValueError(msg)
    return (np.array(replicates), np.array(timepoints),
            len(replicates_counts), len(timepoints_counts))

def is_slice(l):
    return sorted(l) == list(range(min(l), max(l)+1))

def sliceify(indices):
    """
    If a series of indices are consecutive, turn them into a slice to use as and
    index, since this would result in a view (not a copy).
    """
    if len(indices) == 0:
        return indices
    if is_slice(indices):
        return slice(min(indices), max(indices)+1)
    return np.array(indices)

def view_along_axis(arr, indices, axis):
    """
    Equivalent to numpy's np.take_along_axis() but returns a view.
    """
    slices = [slice(None)] * arr.ndim
    slices[axis] = sliceify(indices)
    return arr[tuple(slices)]


def validate_diploids(diploids, R, ntimepoints):
    """
    Since diploids can be a numpy array or integer, this validates the object and returns an
    R x ntimepoints x 1 array that can be broadcast in calculations.
    """
    if isinstance(diploids, int):
        diploids = np.repeat(np.array([diploids], dtype='uint32'), R*ntimepoints)
    elif isinstance(diploids, list):
        # if the reshape fails, something's wrong
        diploids = np.array(diploids, dtype='uint32')

    try:
        diploids = np.array(diploids, dtype='uint32').reshape((R, ntimepoints, 1))
    except:
        fmt = f"diploids must be single integer or array of size R*ntimepoints ({R*ntimepoints})"
        ValueError(fmt)
    # now, try to coerce to smaller sie if possible
    diploids_max = diploids.max()
    if diploids_max < np.iinfo(np.uint8).max:
        diploids = diploids.astype('uint8')
    elif diploids_max < np.iinfo(np.uint16).max:
        diploids = diploids.astype('uint16')
    else:
        diploids = diploids.astype('uint32')
    return diploids
     

def extract_empirical_nulls_diagonals(x, k=0, average_replicates=False, rep=None):
    assert(x.ndim == 5)
    if average_replicates and rep is not None:
        raise ValueError("both average_replicates=True and rep != None")
    if rep is None:
       # swap axes so they are 
        res = np.diagonal(x, offset=k, axis1=2, axis2=3)
        if average_replicates:
            # second to last axis is the replicate axis
            return res.mean(axis=-2)
        return res
    return np.diagonal(x, offset=k, axis1=2, axis2=3)[:, rep, :]

def extract_temporal_cov_diagonals(x, k=0, average_replicates=False, rep=None):
    assert(x.ndim == 4)
    if average_replicates and rep is not None:
        raise ValueError("both average_replicates=True and rep != None")
    if rep is None:
       # swap axes so they are 
        res = np.diagonal(x, offset=k, axis1=1, axis2=2)
        if average_replicates:
            # second to last axis is the replicate axis
            return res.mean(axis=-2)
        return res
    return np.diagonal(x, offset=k, axis1=1, axis2=2)[:, rep, :]


def integerize(x):
    vals = sorted(set(x))
    valmap = {val:i for i, val in enumerate(vals)}
    return [valmap[v] for v in x]

