# --- Runtime Configuration ---
# These are set once at startup and overridden per-dataset by the runner.

import os
import warnings

DIM = 128
NUM_VECTORS = 10000
NUM_QUERIES = 1000
K = 10  # per recall_vs_qps

_physical_cores = os.cpu_count() or 1
_raw_thread_counts = [2, 4, 8, 12, 16, 20, 24]
_capped = [t for t in _raw_thread_counts if t <= _physical_cores]
if len(_capped) < len(_raw_thread_counts):
    _dropped = [t for t in _raw_thread_counts if t > _physical_cores]
    warnings.warn(
        f"THREAD_COUNTS capped at {_physical_cores} physical core(s); "
        f"dropped {_dropped} to avoid context-switching noise.",
        stacklevel=1,
    )
THREAD_COUNTS = _capped or [_physical_cores]


def make_quantized_cls(inner_cls, sq):
    """Wrap an SQ8 index class so it accepts float32 vectors transparently.

    Returns a drop-in replacement for any nilvec index class: the wrapper
    encodes float32 insert/search inputs via *sq* before forwarding to the
    inner int8_t index, so benchmark functions need no changes.
    """

    class QuantizedIndex:
        def __init__(self, dim, *args):
            self._inner = inner_cls(dim, *args)
            self._sq = sq

        def train(self, data):
            encoded = [self._sq.encode(v) for v in data]
            self._inner.train(encoded)

        def insert(self, vec):
            return self._inner.insert(self._sq.encode(vec))

        def search(self, query, k, ef=0):
            encoded = self._sq.encode(query)
            if ef:
                return self._inner.search(encoded, k, ef)
            return self._inner.search(encoded, k)

        def set_nprobe(self, nprobe):
            self._inner.set_nprobe(nprobe)

        def size(self):
            return self._inner.size()

        def max_level(self):
            return self._inner.max_level()

    return QuantizedIndex
