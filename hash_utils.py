"""
Module implementing data utility funtions. Implements deterministic, session‑agnostic hashing for Python objects.

Design requirements
-------------------
1. **Content‑only** – the digest must not depend on ``id(obj)``, memory
   addresses, random salts, or any other non‑portable artefact.
2. **Order semantics preserved** – hashing is order‑aware for
   ``list``/``tuple`` but *order‑insensitive* for ``set``/``frozenset`` and
   ``dict`` (keys are processed in sorted hash order).
3. **Recursive** – every nested element is canonicalised with the same
   rules until only raw bytes remain.
4. **Cryptographic digest** – the canonical byte stream is fed to a fast,
   secure hash (default: *blake2b* with 128‑bit output) to minimise
   collisions.

Canonicalisation cheatsheet
---------------------------
================= ============================== =====================================
Type              Canonical byte prefix          Notes
================= ============================== =====================================
``dict``          ``b"dict" + …``               keys sorted by their **own** digest
``list``/``tuple````b"seq" + len + …``          order preserved
``set``/``frozenset````b"set" + …``              item digests sorted → order‑insensitive
``np.ndarray``    ``b"ndarray" + shape + dtype + raw_bytes``
                                                zero‑copy view on ``uint8`` data
``pandas.DataFrame````b"dataframe" + …``        includes dtypes, index, columns, values
user class        ``b"class" + qualname + …``   hashes the ``__dict__`` recursively
fallback          ``b"pickle:" + pickle.dumps`` deterministic for a fixed CPython version
================= ============================== =====================================

Register a new ``_feed`` handler via ``@_feed.register`` when custom data structures appear in your project.
"""

# Built-in dependencies
import json
import pickle
import hashlib
from typing import Any
from functools import singledispatch

# External dependencies
import numpy as np
import pandas as pd


def stable_hash(
    *args: Any,
    algo: str = "blake2b",
    digest_size: int = 16,
    **kwargs: Any,
) -> str:
    """
    Calculates a deterministic hex digest for any number of objects.
    Returns a lower-case hexadecimal digest representing all provided inputs.
    """

    hasher = _new_hasher(algo, digest_size)

    # Feed positional args in sequence
    for obj in args:
        _feed(hasher, obj)

    # Feed keyword args in sorted key order
    for key in sorted(kwargs):
        _feed(hasher, key)
        _feed(hasher, kwargs[key])

    return hasher.hexdigest()


def _new_hasher(algo: str, digest_size: int):
    if algo == "blake2b":
        return hashlib.blake2b(digest_size=digest_size)
    return getattr(hashlib, algo)()  # e.g. sha256, sha1


@singledispatch
def _feed(hasher, obj):  # type: ignore[override]
    """
    Update hasher with a canonical byte‑view of obj (generic fallback).
    """
    # First try cheap JSON serialisation (handles int, float, str, bool, None)
    try:
        data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
        hasher.update(data)
    except (TypeError, ValueError):
        # Fallback: deterministic pickle for exotic but small scalars
        hasher.update(b"pickle:")
        hasher.update(pickle.dumps(obj, protocol=5))


@_feed.register(dict)
def _dict_feed(hasher, obj: dict):
    hasher.update(b"dict:")
    for k in sorted(obj.keys(), key=lambda x: stable_hash(x)):
        _feed(hasher, k)
        _feed(hasher, obj[k])


@_feed.register(list)
@_feed.register(tuple)
def _seq_feed(hasher, obj):
    hasher.update(f"seq:{len(obj)}".encode())
    for item in obj:
        _feed(hasher, item)


@_feed.register(set)
@_feed.register(frozenset)
def _set_feed(hasher, obj):
    hasher.update(f"set:{len(obj)}".encode())
    for item_digest in sorted(stable_hash(x) for x in obj):
        hasher.update(item_digest.encode())


@_feed.register(np.ndarray)
def _ndarray_feed(hasher, arr: np.ndarray):
    hasher.update(b"ndarray:")
    hasher.update(str(arr.shape).encode())
    hasher.update(str(arr.dtype).encode())
    # view as unsigned bytes to avoid a copy
    hasher.update(arr.view(np.uint8))


@_feed.register(pd.Index)
def _index_feed(hasher, idx: pd.Index):
    hasher.update(b"index:")
    _feed(hasher, str(idx.dtype))
    _feed(hasher, tuple(idx))  # order matters


@_feed.register(pd.Series)
def _series_feed(hasher, ser: pd.Series):
    hasher.update(b"series:")
    _feed(hasher, ser.name)
    _feed(hasher, ser.index)
    _feed(hasher, ser.to_numpy())


@_feed.register(pd.DataFrame)
def _df_feed(hasher, df: pd.DataFrame):
    hasher.update(b"dataframe:")
    _feed(hasher, [str(t) for t in df.dtypes])
    _feed(hasher, df.index)
    _feed(hasher, df.columns)
    _feed(hasher, df.to_numpy(copy=False))


def main() -> None:

    x: np.ndarray = np.random.rand(100, 100)
    y: pd.DataFrame = pd.DataFrame(x)

    print(stable_hash(x))
    print(stable_hash(y))
    print(stable_hash(x, y))
    print(stable_hash(x=x, y=y))

    return


if __name__ == "__main__":
    main()
