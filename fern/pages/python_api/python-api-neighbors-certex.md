---
slug: api-reference/python-api-neighbors-certex
---

# Certex

_Python module: `cuvs.neighbors.certex`_

## Index

```python
cdef class Index
```

CertEx index: exact k-nearest-neighbor search by certified exclusion. Every search returns the
exact k nearest stored vectors of every query.

**Members**

| Name | Kind |
| --- | --- |
| `trained` | property |
| `size` | property |
| `dim` | property |

### trained

```python
def trained(self)
```

Whether the index holds a built index.

### size

```python
def size(self)
```

Number of vectors added (ids 0 to size - 1; removed vectors keep their ids).

### dim

```python
def dim(self)
```

Dimension of the vectors.

## IndexParams

```python
cdef class IndexParams
```

Parameters to build a CertEx index for exact k-nearest-neighbor search.

Every setting returns the same exact neighbors; the settings change only the speed and the
device memory.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | `str, default = "sqeuclidean"` | One of "sqeuclidean", "inner_product", or "cosine". |
| `width` | `int, default = 0` | Projection width m of the half-precision screening rows; 0 chooses the smallest width that explains `variance` of the variance of the dataset. |
| `variance` | `float, default = 0.95` |  |
| `anchor_stride` | `int, default = 16` | Every anchor_stride-th vector is an anchor of the threshold stage. |
| `pca_sample` | `int, default = 1000000` |  |
| `host_vectors` | `bool, default = False` | Keep the full-precision vectors in pinned host memory; the GPU then holds only the screening rows and the anchors. |
| `n_lists` | `int, default = 0` | k-means lists of the vectors; a search first screens the nearest lists of every query, whose exact distances tighten its threshold. |
| `local_projections` | `bool, default = False` | Every list gets its own projection (requires n_lists &gt; 0). |
| `int8_rows` | `bool, default = False` | Screening rows of int8 codes. |

**Constructor**

```python
def __init__(self, *, metric="sqeuclidean", width=0, variance=0.95, anchor_stride=16, pca_sample=1000000, host_vectors=False, n_lists=0, local_projections=False, int8_rows=False)
```

**Members**

| Name | Kind |
| --- | --- |
| `metric` | property |
| `width` | property |
| `anchor_stride` | property |
| `host_vectors` | property |
| `n_lists` | property |
| `local_projections` | property |

### metric

```python
def metric(self)
```

### width

```python
def width(self)
```

### anchor_stride

```python
def anchor_stride(self)
```

### host_vectors

```python
def host_vectors(self)
```

### n_lists

```python
def n_lists(self)
```

### local_projections

```python
def local_projections(self)
```

## SearchParams

```python
cdef class SearchParams
```

Parameters to search a CertEx index.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `anchors` | `int, default = 128` | Anchors per query whose exact distances set the threshold. |
| `batch_size` | `int, default = 2048` | Queries per batch; a batch whose survivors exceed the device memory is split. |
| `probe_lists` | `int, default = 2` | Nearest lists that an index with lists screens first. |

**Constructor**

```python
def __init__(self, *, anchors=128, batch_size=2048, probe_lists=2)
```

**Members**

| Name | Kind |
| --- | --- |
| `anchors` | property |
| `batch_size` | property |
| `probe_lists` | property |

### anchors

```python
def anchors(self)
```

### batch_size

```python
def batch_size(self)
```

### probe_lists

```python
def probe_lists(self)
```

## build

`@auto_sync_resources`

```python
def build(IndexParams index_params, dataset, resources=None)
```

Build a CertEx index from a dataset in device or host memory.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index_params` | `cuvs.neighbors.certex.IndexParams` |  |
| `dataset` | `array interface compliant matrix shape (n_samples, dim)` | Supported dtype [float32, float16, int8, uint8]; device or host memory (a host dataset is streamed to the GPU in chunks). |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.certex.Index` |  |

**Examples**

```python
>>> import cupy as cp
>>> from cuvs.neighbors import certex
>>> dataset = cp.random.random_sample((50000, 64), dtype=cp.float32)
>>> index = certex.build(certex.IndexParams(), dataset)
>>> queries = cp.random.random_sample((1000, 64), dtype=cp.float32)
>>> distances, neighbors = certex.search(certex.SearchParams(), index, queries, 10)
```

## extend

`@auto_sync_resources`

```python
def extend(Index index, new_vectors, resources=None)
```

Append vectors (device or host, the dtype of the index); they receive the next consecutive
ids. Searches may run concurrently from other threads.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.certex.Index` |  |
| `new_vectors` | `array interface compliant matrix shape (n_new, dim)` |  |
| `resources` | `cuvs.common.Resources, optional` |  |

## load

`@auto_sync_resources`

```python
def load(filename, resources=None)
```

Load an index saved by save(); the dtype comes from the file.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `filename` | `str` |  |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.certex.Index` |  |

## remove

`@auto_sync_resources`

```python
def remove(Index index, ids, resources=None)
```

Remove vectors by id; no later search returns them.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `index` | `cuvs.neighbors.certex.Index` |  |
| `ids` | `CUDA array interface compliant vector, dtype int64` |  |
| `resources` | `cuvs.common.Resources, optional` |  |

## save

`@auto_sync_resources`

```python
def save(filename, Index index, resources=None)
```

Save the index, including its vectors, to a file.

The serialization format can change between versions.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `filename` | `str` |  |
| `index` | `cuvs.neighbors.certex.Index` |  |
| `resources` | `cuvs.common.Resources, optional` |  |

## search

`@auto_sync_resources`
`@auto_convert_output`

```python
def search(SearchParams search_params, Index index, queries, k, neighbors=None, distances=None, resources=None, filter=None)
```

Find the exact k nearest neighbors of every query.

**Parameters**

| Name | Type | Description |
| --- | --- | --- |
| `search_params` | `cuvs.neighbors.certex.SearchParams` |  |
| `index` | `cuvs.neighbors.certex.Index` |  |
| `queries` | `CUDA array interface compliant matrix shape (n_queries, dim)` | The dtype of the index. |
| `k` | `int` | The number of neighbors (at most 1024). |
| `neighbors` | `Optional CUDA array interface compliant matrix shape (n_queries, k), dtype int64` |  |
| `distances` | `Optional CUDA array interface compliant matrix shape (n_queries, k), dtype float32` | Squared Euclidean distances, inner products, or cosine distances, as in brute force. |
| `filter` | `Optional cuvs.neighbors.filters filter (bitset over the ids or bitmap per query)` | The result then holds the exact k nearest vectors that the filter keeps (id -1 where fewer exist). |
| `resources` | `cuvs.common.Resources, optional` |  |

**Returns**

distances, neighbors
