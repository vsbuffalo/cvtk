import numpy as np

def integerize(x):
    vals = sorted(set(x))
    valmap = {val:i for i, val in enumerate(vals)}
    return [valmap[v] for v in x]

