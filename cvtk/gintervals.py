"""
GenomicIntervals


"""
import pickle
from itertools import repeat, islice, chain
from operator import itemgetter, attrgetter
from collections import defaultdict, namedtuple, OrderedDict
import numpy as np
import pandas as pd

#from intervaltree import IntervalTree, Interval
from ncls import NCLS

GFFRow = namedtuple('GFFRow', ('seqid', 'source', 'type',
                               'start', 'end', 'score', 'strand',
                               'phase', 'attributes'))

Interval = namedtuple('Interval', ('start', 'end', 'data'))

    # def build_ncls(self):
    #     ncls = dict()
    #     for seqid, intervals in self.intervals.items():
    #         starts = [i.start for i in intervals]
    #         ends = [i.end for i in intervals]
    #         indices = [i.data for i in intervals]
    #         ncls[seqid] = NestedContainmentList(starts, ends, indices)
    #     return ncls

def overlaps(start_a, end_a, start_b, end_b):
    return start_a < end_b and end_a > start_b

def view_along_axis(arr, indices, axis):
    """
    Equivalent to numpy's np.take_along_axis() but returns a view.
    """
    slices = [slice(None)] * arr.ndim
    slices[axis] = sliceify(indices)
    return arr[tuple(slices)]


def weighted_average(vec, weights, finite_only=True):
    vec = np.array(vec)
    weights = np.array(weights)
    if finite_only:
        valid = np.logical_and(np.isfinite(vec), np.isfinite(weights))
        vec = vec[valid]
        weights = weights[valid]
    if len(weights) == 0 or len(weights) == 0:
        return np.nan
    return np.average(vec, weights=weights)

def merge_overlaps(intervals, data_func=max):
    """
    O(log n) complexity
    """
    sorted_by_start = sorted(intervals, key=itemgetter(0))
    merged = []
    for interval in sorted_by_start:
        if not merged:
            merged.append(interval)
        else:
            last_interval = merged[-1]
            if last_interval[1] >= interval[0]:
                new_end = max(interval[1], last_interval[1])
                new_data = data_func(interval[2], last_interval[2])
                merged[-1] = (last_interval[0], new_end, new_data)
            else:
                merged.append(interval)
    return merged


class NestedContainmentList(object):
    def __init__(self, starts=None, ends=None, indices=None, reduce=False):
        self.ncls = None
        if starts is not None and indices is not None:
            if ends is None:
                ends = [s + 1 for s in starts]
            if reduce:
                starts, ends, indices = list(zip(*merge_overlaps(zip(starts, ends, indices))))
            starts = np.array(starts, dtype='i8')
            ends = np.array(ends, dtype='i8')
            indices = np.array(indices, dtype='i8')
            self.ncls = NCLS(starts, ends, indices)

    def find_overlaps(self, start, end):
        if self.ncls is None:
            # we allow for empty objects, in which case nothing overlaps
            # use case: non-matching seqids
            return []
        overlaps = []
        for overlap in self.ncls.find_overlap(start, end):
            overlaps.append(Interval(*overlap))
        return overlaps

    @staticmethod
    def from_intervals(intervals, reduce=False):
        starts, ends, indices = zip(*intervals)
        starts = np.array(starts, dtype='i8')
        ends = np.array(ends, dtype='i8')
        indices = np.array(indices, dtype='i8')
        obj = NestedContainmentList(starts, ends, indices, reduce=reduce)
        return obj


def _parse_gff_attributes(attrs_str, keyval_sep, keyval_delim):
    try:
        keyvals = attrs_str.strip(";").split(keyval_delim)
        attrs = dict()
        for item in keyvals:
            key, val = item.split(keyval_sep)
            val = val.strip('"')
            attrs[key] = val
    except ValueError:
        msg = (f"error parsing GFF/GTF, check that attribute delimiters "
               "specified correctly")
        raise ValueError(msg)
        return attrs

def read_gff(filename, delims=('=', ';')):
    """
    A permissive GFF/GTF parser.
    """
    try:
        tag_value_delim, attribute_delim = delims
    except ValueError:
        msg = "delims must be a tuple of (tag-value, attribute) delimiters."
        raise ValueError(msg)
    with open(filename) as gff:
        for line in gff:
            if line.startswith('#'):
                continue
            fields = line.strip().split("\t")
            assert(len(fields) == 9)
            seqid, source, type = fields[0:3]
            start, end = map(int, fields[3:5])
            score = float(fields[5]) if fields[5] is not '.' else float('nan')
            strand, phase = fields[6:8]
            attr = _parse_gff_attributes(fields[8], tag_value_delim, attribute_delim)
            yield GFFRow(seqid, source, type, start, end, score, strand, phase, attr)

def group_gff_by_type(iter):
    """
    Take an iterable of rows of GFF file and group by type (previously known as
    feature).  Returns a defaultdict(list) of group->[GTF rows].
    """
    grouped = OrderedDict()
    for row in iter:
        if row.type not in grouped:
            grouped[row.type] = list()
        grouped[row.type].append(row)
    return grouped


class OrderedDictList(OrderedDict):
    def __init__(self):
        self.default_factory = list
        super().__init__()

    def __missing__(self, key):
        # this mimics defaultdict
        value = self.default_factory()
        self[key] = value
        return value

class GenomicInterval(object):
    __slots__ = ('seqid', 'interval')
    def __init__(self, seqid, start, end=None, data=None):
        if end == None:
            # assume a single bp
            end = start + 1
        assert(end > start)
        self.seqid = seqid
        self.interval = Interval(start, end, data)

    def __eq__(self, other):
        # this does not compare data!
        is_same = (self.seqid == other.seqid and
                   self.interval[0] == other.interval[0] and
                   self.interval[1] == other.interval[1])
        return is_same

    def __repr__(self):
        return f"GenomicInterval({self.seqid}, [{self.start}, {self.end}))"

    def __str__(self):
        return f"{self.seqid}:[{self.start}, {self.end})"

    @property
    def data(self):
        return self.interval.data

    @property
    def start(self):
        return self.interval.start

    @property
    def end(self):
        return self.interval.end

    @property
    def width(self):
        return self.end - self.start

    def overlaps(self, interval):
        if self.seqid != interval.seqid:
            return False
        return overlaps(self.start, self.end, interval.start, interval.end)


class GenomicIntervals(object):
    """
    Maintains insertion order.
    """
    def __init__(self, intervals=None, seqlens=None):
        self.intervals = OrderedDictList()
        self.data = pd.DataFrame()
        self.seqlens = seqlens
        self._idx = 0
        self._index = []
        if intervals is not None:
            self.extend(intervals)

#     def get_data(self, index):
#         if isinstance(self.data, pd.DataFrame):
#             return self.data.loc[index]
#         if isinstance(self, np.ndarray):
#             return view_along_axis(self.data, index, axis=0)
#         if isinstance(self. list):
#             self.data[index]
#         msg = ("GenomicIntervals.data must be either a pandas DataFrame, "
#                "numpy ndarray, or list")
#         raise ValueError(msg)

    def set_data(self, data):
        assert(isinstance(data, (list, pd.DataFrame)))
        if isinstance(data, pd.DataFrame):
            assert(data.shape[0] == len(self))
        if isinstance(data, list):
            assert(len(data) == len(self))
        self.data = pd.DataFrame(data, index=self._index)

    def grow(self, buffer_start, buffer_end):
        """
        Grow the intervals to [buffer_start - start, buffer_end).
        """
        for interval in iter(self):
            new_start = max(0, interval.start - buffer_start)
            new_end = min(buffer_end + buffer_end, self.seqlens[interval.seqid])
            interval.grow(new_start, new_end)

    def append_interval(self, interval):
        self.append(interval.seqid, interval.start, interval.end)

    def append(self, seqid, start, end=None, check_bounds=True):
        end = end if end is not None else start + 1
        if check_bounds and self.seqlens is not None:
            max_pos = self.seqlens[seqid]
            out_of_bounds = (start >= max_pos or end > max_pos)
            if out_of_bounds:
                fmt = (f"interval [{start}, {end}) out of bounds ({seqid} "
                       f"length = {max_pos}")
                raise ValueError(fmt)
        interval = GenomicInterval(seqid, start, end, data=self._idx)
        self.intervals[seqid].append(interval)
        self._index.append(self._idx)
        self._idx += 1

    def extend(self, intervals):
        for seqid, start, end in intervals:
            self.append(seqid, start, end)

    def keys(self):
        return self.intervals.keys()

    @property
    def seqid(self):
        return [i.seqid for i in chain(*self.intervals.values())]

    @property
    def midpoint(self):
        return [(i.start + i.end)/2 for i in chain(*self.intervals.values())]


    @property
    def cummulative_midpoint(self):
        if self.seqlens is None:
            msg = "seqlens attribute must be set to calculate cummulative midpoint"
            raise ValueError(msg)
        # build up cumulative seqlens
        seqlens_cumsum = np.cumsum([0] + [self.seqlens[seqid] for seqid in self.keys()][:-1])
        offsets = dict(zip(self.keys(), seqlens_cumsum))
        return [offsets[i.seqid] + (i.start + i.end)/2 for i in chain(*self.intervals.values())]

    @property
    def start(self):
        return [i.start for i in chain(*self.intervals.values())]

    @property
    def end(self):
        return [i.end for i in chain(*self.intervals.values())]


    def __getitem__(self, seqid):
        return self.intervals[seqid]

    def __iter__(self):
        for intervals in self.intervals.values():
            yield from intervals

    def itertuples(self):
        for interval in self:
            yield (interval.seqid, interval)

    def infer_seqlens(self):
        endpoints = defaultdict(int)
        for interval in self:
            endpoints[interval.seqid] = max(endpoints[interval.seqid], interval.end)
        self.seqlens = endpoints

    @property
    def nseqids(self):
        return len(self.keys())

    def __len__(self):
        return sum(len(v) for v in self.intervals.values())

    def _head(self, seqid=None, nentries=10):
        i = 0
        head = []
        header = f"GenomicIntervals — {self.nseqids} unique seqids, {len(self)} features\n"
        intervals = list(islice(self, nentries))
        head = pd.DataFrame(dict(GenomicInterval=intervals))
        footer = ""
        if len(self) > nentries:
            footer = f"\n[ {len(self) - nentries} more GenomicIntervals ]"
        table = pd.concat([head, self.data], axis=1).head(nentries).to_string()
        return header + table + footer

    def head(self, seqid=None, nentries=10):
        print(self._head(nentries=nentries))

    def __repr__(self):
       return self._head(nentries=5)

    def build_ncls(self, reduce=False):
        ncls = defaultdict(NestedContainmentList)
        for seqid, intervals in self.intervals.items():
            ivals = map(attrgetter('interval'), intervals)
            ncls[seqid] = NestedContainmentList.from_intervals(ivals, reduce=reduce)
        return ncls

    @staticmethod
    def from_copy(original_object):
        obj = GenomicIntervals()
        original_object.seqlens = obj.seqlens
        for seqid, interval in original_object.itertuples():
            # this isn't a deep copy down to data level
            obj.append(interval.seqid, interval.start, interval.end)
        obj.seqlens = original_object.seqlens
        obj.data = original_object.data.copy()
        return obj

    def nest_by_overlaps(self, intervals_to_group, colname=None,
                         nest='interval'):
        """
        """
        if nest not in ('index', 'interval', 'data'):
            raise ValueError("nest must be either 'index', 'interval', 'data'")
        if not isinstance(intervals_to_group, GenomicIntervals):
            msg = "intervals_to_group must be a GenomicIntervals object"
            raise ValueError(msg)
        query_ncls = intervals_to_group.build_ncls()
        groups = self
        column = []
        for seqid, interval in groups.itertuples():
            olaps = sorted(query_ncls[seqid].find_overlaps(interval.start, interval.end))
            to_nest = list()
            for olap in olaps:
                # we remove the index, used for the interval tree
                if nest == 'data':
                    try:
                        item = intervals_to_group.data.loc[olap.data]
                    except KeyError:
                        msg = f"no data found in intervals_to_group at key {olap.data}"
                        raise KeyError(msg)
                elif nest == 'index':
                    item = olap.data
                else:
                    item = GenomicInterval(seqid, olap.start, olap.end, olap.data)
                to_nest.append(item)
            column.append(to_nest)
        if colname is not None:
            self.data[colname] = column
        else:
            return column
        return self

    def split_array_by_overlaps(self, intervals_to_group, array, axis,
                                colname=None):
        """
        Through view_along_axis(), this will try to return a view.
        """
        index_groups = self.nest_by_overlaps(intervals_to_group, nest='index',
                                        as_column=False)
        views = list()
        for indices in index_groups:
            views.append(view_along_axis(array, indices, axis))
        if colname is not None:
            self.data[colname] = views
        return views

    #def find_overlaps(self, query_interval):
    #    """
    #    Return the indices of intervals that overlap query interval.
    #    """
    #    for i, (seqid, interval) in enumerate(self.itertuples()):
    #        if interval.overlaps(query_interval):
    #            yield i

    def overlap_coverage(self, intervals, colname=None,
                         proportion=False, reduce=True):
        """
        Find the overlaps between this object and the GenomicIntervals argument
        'intervals', and calculate the coverage. If reduce=True,
        IntervalTree.merge_overlaps() is run on 'intervals' first. If
        proportion=True, the proportion of overlapping bases is returned (note
        that unless reduce=True, this can be > 1).

        A new column with name 'colname' will be added to GenomicIntervals.data.
        """
        ncls = intervals.build_ncls(reduce=reduce)
        # clear out old coverage data
        column = []
        for seqid, interval in self.itertuples():
            olaps = ncls[seqid].find_overlaps(interval.start, interval.end)
            coverage = 0
            for olap in olaps:
                start = interval.start if olap.start < interval.start else olap.start
                end = interval.end if olap.end > interval.end else olap.end
                cov = end - start
                if proportion:
                    coverage += cov/(interval.end - interval.start)
                else:
                    coverage += cov
            column.append(coverage)
        if colname is not None:
            self.data[colname] = column
        else:
            return column
        return self


#     def get_overlaps(self, chr, start, end, retval='intervals'):
#         ncls = NestedContainmentList.from_intervals(self.intervals[chr])
#         olaps = ncls.find_overlaps(start, end)
#         if retval == 'intervals':
#             return olaps
#         if retval == 'indices':
#             return [o.data for o in olaps]
#         raise ValueError("retval must be 'intervals' or 'indices'")


    def stat_by_coverage(self, intervals, stat_colname,
                         stat_fun=weighted_average, colname=None,
                         pass_weights=True):
        """
        TODO HERE
        """
        ncls = intervals.build_ncls()
        # clear out old coverage data
        column = []
        for seqid, interval in self.itertuples():
            olaps = ncls[seqid].find_overlaps(interval.start, interval.end)
            vector = []  # the vector to apply the statistic on
            weights = []  # the coverage weights
            for olap in olaps:
                start = interval.start if olap.start < interval.start else olap.start
                end = interval.end if olap.end > interval.end else olap.end
                # this is the coverage of the current query overlap over the
                # current interval
                cov = end - start
                weight = cov / (interval.end - interval.start)
                weights.append(weight)
                vector.append(intervals.data[stat_colname].loc[olap.data])
            assert(len(vector) == len(weights))
            if not len(vector):
                #import pdb; pdb.set_trace()
                column.append(np.nan)
            else:
                if pass_weights:
                    stat = stat_fun(vector, weights)
                else:
                    stat = stat_fun(vector)
                column.append(stat)
        if colname is not None:
            self.data[colname] = column
        else:
            return column
        return self

    @staticmethod
    def from_gff(filename, filter_type=None, filter_seqids=None, delims=('=', ';')):
        obj = GenomicIntervals()
        keep_types = set()
        if filter_type is not None:
            filter_type = [filter_type] if isinstance(filter_type, str) else filter_type
            keep_types = set(filter_type)
        if filter_seqids is not None:
            filter_seqids = [filter_seqids] if isinstance(filter_seqids, str) else filter_seqids
            keep_seqids = set(filter_seqids)
        df_rows = list()
        for feature in read_gff(filename, delims):
            if filter_type is not None and feature.type not in keep_types:
                continue
            if filter_seqids is not None and feature.seqid not in keep_seqids:
                continue
            # GFF are 1-indexed, so we offset start
            obj.append(feature.seqid, feature.start - 1, feature.end)
            row = dict(**{k: v for k, v in feature._asdict().items()
                                if k not in ('start', 'end')})
            df_rows.append(row)
        obj.set_data(df_rows)
        return obj

    @staticmethod
    def from_tiles(seqlens, width, end_buffer=0, drop_last_tile=True):
        """
        create bins of a particular physical (e.g. megabases) width.
        """
        ## create tile intervals
        obj = GenomicIntervals(seqlens=seqlens)
        for seqid, last_position in obj.seqlens.items():
            # use, or do not use the last tile accordingly
            rounder_fun = np.ceil if not drop_last_tile else np.floor
            endpoint = rounder_fun(last_position / width) * width
            binpoints = np.arange(0 + end_buffer, endpoint - end_buffer, width, dtype='u4')
            for binpoint in binpoints:
                obj.append(seqid, binpoint, binpoint + width, check_bounds=False)
        obj.data = pd.DataFrame(index=obj._index)
        return obj

    def __eq__(self, other):
        same_seqids = self.intervals.keys() == other.intervals.keys()
        both_intervals = zip(self.intervals.values(), other.intervals.values())
        same_intervals = all(ours == theirs for ours, theirs in both_intervals)
        same_data = all(self.data == other.data)
        return same_data and same_intervals and same_seqids

    def to_df(self, add_midpoint=False, add_cummidpoint=False):
        gidf = OrderedDict(seqid=self.seqid, start=self.start, end=self.end)
        if add_midpoint:
            gidf['midpoint'] = (np.array(gidf['start']) + np.array(gidf['end']))/2
        if add_cummidpoint:
            gidf['cummidpoint'] = self.cummulative_midpoint
        return pd.concat([pd.DataFrame(gidf), self.data], axis=1)

    def dump(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file):
        with open(file, 'rb') as f:
            obj = pickle.load(f)
        return obj

