# SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# Tests of cuvs.neighbors.certex against exhaustive search in double precision, in the style of the
# cuVS tests (numpy references, pylibraft device arrays).

import cupy as cp
import numpy as np
import pytest
from pylibraft.common import device_ndarray

from cuvs.neighbors import certex, filters

pytestmark = pytest.mark.skipif(
    int(cp.cuda.Device().compute_capability) < 80,
    reason="CertEx needs a GPU of compute capability 8.0 or later",
)


def scores(X, Q, metric):
    """Exact scores in double precision (smaller is nearer)."""
    Xd, Qd = X.astype(np.float64), Q.astype(np.float64)
    if metric == "sqeuclidean":
        return (
            (Qd * Qd).sum(1)[:, None]
            + (Xd * Xd).sum(1)[None, :]
            - 2.0 * Qd @ Xd.T
        )
    if metric == "inner_product":
        return -(Qd @ Xd.T)
    norms = np.linalg.norm(Qd, axis=1)[:, None] * np.linalg.norm(Xd, axis=1)
    return 1.0 - (Qd @ Xd.T) / norms


def check_exact(X, Q, indices, k, metric="sqeuclidean", keep=None):
    """The returned neighbors carry the k smallest exact scores."""
    s = scores(X, Q, metric)
    if keep is not None:
        s[:, ~keep] = np.inf
    ref = np.sort(s, axis=1)[:, :k]
    got = np.sort(np.take_along_axis(s, indices.astype(np.int64), 1), 1)
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9)


def random_data(n=20000, d=32, nq=200, seed=0, dtype=np.float32):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(dtype)
    Q = rng.standard_normal((nq, d)).astype(dtype)
    return X, Q


@pytest.mark.parametrize("metric", ["sqeuclidean", "inner_product", "cosine"])
@pytest.mark.parametrize(
    "n_lists,local", [(0, False), (32, False), (32, True)]
)
@pytest.mark.parametrize("k", [1, 10, 64])
def test_certex_search(metric, n_lists, local, k):
    X, Q = random_data()
    params = certex.IndexParams(
        metric=metric, width=16, n_lists=n_lists, local_projections=local
    )
    index = certex.build(params, device_ndarray(X))
    assert index.trained
    assert index.size == X.shape[0] and index.dim == X.shape[1]
    _, neighbors = certex.search(
        certex.SearchParams(), index, device_ndarray(Q), k
    )
    neighbors = neighbors.copy_to_host()
    assert neighbors.shape == (Q.shape[0], k)
    check_exact(X, Q, neighbors, k, metric)


def test_certex_preallocated_outputs():
    X, Q = random_data(seed=4)
    k = 10
    index = certex.build(certex.IndexParams(width=16), device_ndarray(X))
    neighbors = device_ndarray(np.zeros((Q.shape[0], k), dtype=np.int64))
    distances = device_ndarray(np.zeros((Q.shape[0], k), dtype=np.float32))
    certex.search(
        certex.SearchParams(),
        index,
        device_ndarray(Q),
        k,
        neighbors=neighbors,
        distances=distances,
    )
    found = neighbors.copy_to_host()
    check_exact(X, Q, found, k)
    D = distances.copy_to_host()
    ref = np.take_along_axis(scores(X, Q, "sqeuclidean"), found, 1)
    np.testing.assert_allclose(D, ref, rtol=1e-4, atol=1e-3)


def test_certex_bitset_filter():
    X, Q = random_data(seed=1)
    rng = np.random.default_rng(1)
    keep = rng.random(X.shape[0]) < 0.3
    padded = np.zeros(-(-X.shape[0] // 32) * 32, dtype=bool)
    padded[: X.shape[0]] = keep
    bits = np.packbits(padded.reshape(-1, 32)[:, ::-1], axis=1)
    bitset = bits.view(">u4").astype(np.uint32).ravel()
    index = certex.build(certex.IndexParams(width=16), device_ndarray(X))
    _, neighbors = certex.search(
        certex.SearchParams(),
        index,
        device_ndarray(Q),
        10,
        filter=filters.from_bitset(device_ndarray(bitset)),
    )
    found = neighbors.copy_to_host()
    assert keep[found].all()
    check_exact(X, Q, found, 10, keep=keep)


def test_certex_extend_remove_host_save_load(tmp_path):
    X, Q = random_data(seed=2)
    index = certex.build(
        certex.IndexParams(width=16, host_vectors=True), X[:10000]
    )
    certex.extend(index, device_ndarray(X[10000:]))
    assert index.size == X.shape[0]
    gone = np.arange(0, X.shape[0], 7, dtype=np.int64)
    certex.remove(index, device_ndarray(gone))
    keep = np.ones(X.shape[0], dtype=bool)
    keep[gone] = False
    _, neighbors = certex.search(
        certex.SearchParams(), index, device_ndarray(Q), 10
    )
    check_exact(X, Q, neighbors.copy_to_host(), 10, keep=keep)
    path = str(tmp_path / "index.certex")
    certex.save(path, index)
    loaded = certex.load(path)
    _, neighbors = certex.search(
        certex.SearchParams(), loaded, device_ndarray(Q), 10
    )
    check_exact(X, Q, neighbors.copy_to_host(), 10, keep=keep)


@pytest.mark.parametrize("dtype", [np.float16, np.uint8, np.int8])
def test_certex_dtypes(dtype):
    rng = np.random.default_rng(3)
    if dtype == np.float16:
        X, Q = random_data(d=64, nq=100, seed=3, dtype=np.float16)
    else:
        lo, hi = (0, 256) if dtype == np.uint8 else (-128, 128)
        X = rng.integers(lo, hi, (20000, 64)).astype(dtype)
        Q = rng.integers(lo, hi, (100, 64)).astype(dtype)
    index = certex.build(certex.IndexParams(width=24), device_ndarray(X))
    _, neighbors = certex.search(
        certex.SearchParams(), index, device_ndarray(Q), 10
    )
    check_exact(X, Q, neighbors.copy_to_host(), 10)
