# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

import numpy as np

cimport cuvs.common.cydlpack

from cuvs.common.resources import auto_sync_resources

from libc.stdint cimport int64_t, uint32_t
from libcpp cimport bool
from libcpp.string cimport string

from cuvs.common cimport cydlpack
from cuvs.distance_type cimport cuvsDistanceType

from pylibraft.common import auto_convert_output, device_ndarray
from pylibraft.common.cai_wrapper import wrap_array
from pylibraft.common.interruptible import cuda_interruptible

from cuvs.distance import DISTANCE_TYPES
from cuvs.neighbors.common import _check_input_array

from cuvs.common.c_api cimport cuvsResources_t

from cuvs.common.exceptions import check_cuvs
from cuvs.neighbors.filters import no_filter

_DTYPES = [np.dtype("float32"), np.dtype("float16"), np.dtype("int8"), np.dtype("uint8")]


cdef class IndexParams:
    """
    Parameters to build a CertEx index for exact k-nearest-neighbor search.

    Every setting returns the same exact neighbors; the settings change only the speed and the
    device memory.

    Parameters
    ----------
    metric : str, default = "sqeuclidean"
        One of "sqeuclidean", "inner_product", or "cosine".
    width : int, default = 0
        Projection width m of the half-precision screening rows; 0 chooses the smallest width that
        explains `variance` of the variance of the dataset.
    variance : float, default = 0.95
    anchor_stride : int, default = 16
        Every anchor_stride-th vector is an anchor of the threshold stage.
    pca_sample : int, default = 1000000
    host_vectors : bool, default = False
        Keep the full-precision vectors in pinned host memory; the GPU then holds only the
        screening rows and the anchors.
    n_lists : int, default = 0
        k-means lists of the vectors; a search first screens the nearest lists of every query,
        whose exact distances tighten its threshold.
    local_projections : bool, default = False
        Every list gets its own projection (requires n_lists > 0).
    int8_rows : bool, default = False
        Screening rows of int8 codes.
    """

    cdef cuvsCertexIndexParams* params
    cdef object _metric

    def __cinit__(self):
        check_cuvs(cuvsCertexIndexParamsCreate(&self.params))

    def __dealloc__(self):
        if self.params is not NULL:
            check_cuvs(cuvsCertexIndexParamsDestroy(self.params))

    def __init__(self, *,
                 metric="sqeuclidean",
                 width=0,
                 variance=0.95,
                 anchor_stride=16,
                 pca_sample=1000000,
                 host_vectors=False,
                 n_lists=0,
                 local_projections=False,
                 int8_rows=False):
        if metric not in ("sqeuclidean", "inner_product", "cosine"):
            raise ValueError("metric must be sqeuclidean, inner_product, or cosine")
        self._metric = metric
        self.params.metric = <cuvsDistanceType>DISTANCE_TYPES[metric]
        self.params.width = width
        self.params.variance = variance
        self.params.anchor_stride = anchor_stride
        self.params.pca_sample = pca_sample
        self.params.host_vectors = host_vectors
        self.params.n_lists = n_lists
        self.params.local_projections = local_projections
        self.params.int8_rows = int8_rows

    @property
    def metric(self):
        return self._metric

    @property
    def width(self):
        return self.params.width

    @property
    def anchor_stride(self):
        return self.params.anchor_stride

    @property
    def host_vectors(self):
        return self.params.host_vectors

    @property
    def n_lists(self):
        return self.params.n_lists

    @property
    def local_projections(self):
        return self.params.local_projections


cdef class SearchParams:
    """
    Parameters to search a CertEx index.

    Parameters
    ----------
    anchors : int, default = 128
        Anchors per query whose exact distances set the threshold.
    batch_size : int, default = 2048
        Queries per batch; a batch whose survivors exceed the device memory is split.
    probe_lists : int, default = 2
        Nearest lists that an index with lists screens first.
    """

    cdef cuvsCertexSearchParams* params

    def __cinit__(self):
        check_cuvs(cuvsCertexSearchParamsCreate(&self.params))

    def __dealloc__(self):
        if self.params is not NULL:
            check_cuvs(cuvsCertexSearchParamsDestroy(self.params))

    def __init__(self, *, anchors=128, batch_size=2048, probe_lists=2):
        self.params.anchors = anchors
        self.params.batch_size = batch_size
        self.params.probe_lists = probe_lists

    @property
    def anchors(self):
        return self.params.anchors

    @property
    def batch_size(self):
        return self.params.batch_size

    @property
    def probe_lists(self):
        return self.params.probe_lists


cdef class Index:
    """
    CertEx index: exact k-nearest-neighbor search by certified exclusion. Every search returns the
    exact k nearest stored vectors of every query.
    """

    cdef cuvsCertexIndex_t index
    cdef bool trained

    def __cinit__(self):
        self.trained = False
        check_cuvs(cuvsCertexIndexCreate(&self.index))

    def __dealloc__(self):
        if self.index is not NULL:
            check_cuvs(cuvsCertexIndexDestroy(self.index))

    @property
    def trained(self):
        """Whether the index holds a built index."""
        return self.trained

    @property
    def size(self):
        """Number of vectors added (ids 0 to size - 1; removed vectors keep their ids)."""
        cdef int64_t n = 0
        if self.trained:
            check_cuvs(cuvsCertexIndexGetSize(self.index, &n))
        return n

    @property
    def dim(self):
        """Dimension of the vectors."""
        cdef int64_t d = 0
        if self.trained:
            check_cuvs(cuvsCertexIndexGetDim(self.index, &d))
        return d

    def __repr__(self):
        return f"Index(type=CertEx, size={self.size}, dim={self.dim})"


@auto_sync_resources
def build(IndexParams index_params, dataset, resources=None):
    """
    Build a CertEx index from a dataset in device or host memory.

    Parameters
    ----------
    index_params : cuvs.neighbors.certex.IndexParams
    dataset : array interface compliant matrix shape (n_samples, dim)
        Supported dtype [float32, float16, int8, uint8]; device or host memory (a host dataset is
        streamed to the GPU in chunks).
    {resources_docstring}

    Returns
    -------
    index: cuvs.neighbors.certex.Index

    Examples
    --------

    >>> import cupy as cp
    >>> from cuvs.neighbors import certex
    >>> dataset = cp.random.random_sample((50000, 64), dtype=cp.float32)
    >>> index = certex.build(certex.IndexParams(), dataset)
    >>> queries = cp.random.random_sample((1000, 64), dtype=cp.float32)
    >>> distances, neighbors = certex.search(certex.SearchParams(), index, queries, 10)
    """
    dataset_ai = wrap_array(dataset)
    _check_input_array(dataset_ai, _DTYPES)

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef Index idx = Index()
    cdef cydlpack.DLManagedTensor* dataset_dlpack = cydlpack.dlpack_c(dataset_ai)

    with cuda_interruptible():
        check_cuvs(cuvsCertexBuild(res, index_params.params, dataset_dlpack, idx.index))
        idx.trained = True
    return idx


@auto_sync_resources
def extend(Index index, new_vectors, resources=None):
    """
    Append vectors (device or host, the dtype of the index); they receive the next consecutive
    ids. Searches may run concurrently from other threads.

    Parameters
    ----------
    index : cuvs.neighbors.certex.Index
    new_vectors : array interface compliant matrix shape (n_new, dim)
    {resources_docstring}
    """
    if not index.trained:
        raise ValueError("Index needs to be built before calling extend.")
    vectors_ai = wrap_array(new_vectors)
    _check_input_array(vectors_ai, _DTYPES)
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef cydlpack.DLManagedTensor* vectors_dlpack = cydlpack.dlpack_c(vectors_ai)
    with cuda_interruptible():
        check_cuvs(cuvsCertexExtend(res, vectors_dlpack, index.index))
    return index


@auto_sync_resources
def remove(Index index, ids, resources=None):
    """
    Remove vectors by id; no later search returns them.

    Parameters
    ----------
    index : cuvs.neighbors.certex.Index
    ids : CUDA array interface compliant vector, dtype int64
    {resources_docstring}
    """
    if not index.trained:
        raise ValueError("Index needs to be built before calling remove.")
    ids_ai = wrap_array(ids)
    _check_input_array(ids_ai, [np.dtype("int64")])
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef cydlpack.DLManagedTensor* ids_dlpack = cydlpack.dlpack_c(ids_ai)
    with cuda_interruptible():
        check_cuvs(cuvsCertexRemove(res, ids_dlpack, index.index))
    return index


@auto_sync_resources
@auto_convert_output
def search(SearchParams search_params,
           Index index,
           queries,
           k,
           neighbors=None,
           distances=None,
           resources=None,
           filter=None):
    """
    Find the exact k nearest neighbors of every query.

    Parameters
    ----------
    search_params : cuvs.neighbors.certex.SearchParams
    index : cuvs.neighbors.certex.Index
    queries : CUDA array interface compliant matrix shape (n_queries, dim)
        The dtype of the index.
    k : int
        The number of neighbors (at most 1024).
    neighbors : Optional CUDA array interface compliant matrix shape (n_queries, k), dtype int64
    distances : Optional CUDA array interface compliant matrix shape (n_queries, k), dtype float32
        Squared Euclidean distances, inner products, or cosine distances, as in brute force.
    filter : Optional cuvs.neighbors.filters filter (bitset over the ids or bitmap per query)
        The result then holds the exact k nearest vectors that the filter keeps (id -1 where
        fewer exist).
    {resources_docstring}

    Returns
    -------
    distances, neighbors
    """
    if not index.trained:
        raise ValueError("Index needs to be built before calling search.")

    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    queries_cai = wrap_array(queries)
    _check_input_array(queries_cai, _DTYPES)
    cdef uint32_t n_queries = queries_cai.shape[0]

    if neighbors is None:
        neighbors = device_ndarray.empty((n_queries, k), dtype="int64")
    neighbors_cai = wrap_array(neighbors)
    _check_input_array(neighbors_cai, [np.dtype("int64")], exp_rows=n_queries, exp_cols=k)

    if distances is None:
        distances = device_ndarray.empty((n_queries, k), dtype="float32")
    distances_cai = wrap_array(distances)
    _check_input_array(distances_cai, [np.dtype("float32")], exp_rows=n_queries, exp_cols=k)

    cdef cydlpack.DLManagedTensor* queries_dlpack = cydlpack.dlpack_c(queries_cai)
    cdef cydlpack.DLManagedTensor* neighbors_dlpack = cydlpack.dlpack_c(neighbors_cai)
    cdef cydlpack.DLManagedTensor* distances_dlpack = cydlpack.dlpack_c(distances_cai)

    if filter is None:
        filter = no_filter()

    with cuda_interruptible():
        check_cuvs(cuvsCertexSearch(res,
                                    search_params.params,
                                    index.index,
                                    queries_dlpack,
                                    neighbors_dlpack,
                                    distances_dlpack,
                                    filter.prefilter))
    return (distances, neighbors)


@auto_sync_resources
def save(filename, Index index, resources=None):
    """
    Save the index, including its vectors, to a file.

    The serialization format can change between versions.

    Parameters
    ----------
    filename : str
    index : cuvs.neighbors.certex.Index
    {resources_docstring}
    """
    if not index.trained:
        raise ValueError("Index needs to be built before calling save.")
    cdef string c_filename = filename.encode("utf-8")
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    check_cuvs(cuvsCertexSerialize(res, c_filename.c_str(), index.index))


@auto_sync_resources
def load(filename, resources=None):
    """
    Load an index saved by save(); the dtype comes from the file.

    Parameters
    ----------
    filename : str
    {resources_docstring}

    Returns
    -------
    index : cuvs.neighbors.certex.Index
    """
    cdef Index idx = Index()
    cdef cuvsResources_t res = <cuvsResources_t>resources.get_c_obj()
    cdef string c_filename = filename.encode("utf-8")
    check_cuvs(cuvsCertexDeserialize(res, c_filename.c_str(), idx.index))
    idx.trained = True
    return idx
