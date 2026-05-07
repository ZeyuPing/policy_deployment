"""msgpack codec with numpy ndarray support.

Why msgpack:
- secure (no arbitrary code execution, unlike pickle)
- schema-less, fast, ~4x smaller/faster than pickle for large arrays
- cross-language

The encoding uses byte-string keys (b"__ndarray__", b"data", ...) so that
both ends decode identically regardless of msgpack's `raw` flag.
"""

from __future__ import annotations

import functools

import msgpack
import numpy as np


def pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)
Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
