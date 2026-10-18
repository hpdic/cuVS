---
slug: api-reference/cpp-api-neighbors-certex
---

# Certex

_Source header: `cuvs/neighbors/certex.hpp`_

## CertEx index build parameters

<a id="neighbors-certex-index-params"></a>
### neighbors::certex::index_params

Parameters of a CertEx index.

Supported metrics: L2Expanded, InnerProduct, and CosineExpanded.

```cpp
struct index_params : cuvs::neighbors::index_params {
  int width;
  double variance;
  int anchor_stride;
  int64_t pca_sample;
  bool host_vectors;
  int n_lists;
  int64_t kmeans_sample;
  bool local_projections;
  bool int8_rows;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `width` | `int` | Number m of principal directions kept in the half-precision screening rows. 0 picks the smallest tile-aligned width that captures `variance` of the spectrum. Every width returns the exact neighbors; the width only changes the speed and the device memory per vector. |
| `variance` | `double` | Fraction of the variance that an automatic width captures. |
| `anchor_stride` | `int` | Every anchor_stride-th vector is an anchor, whose exact distance bounds the k-th distance. |
| `pca_sample` | `int64_t` | Number of vectors that the principal directions are computed from. |
| `host_vectors` | `bool` | Keep the full-precision vectors in pinned host memory instead of device memory. Device memory then holds only the screening rows and the anchors (about 2 (m + 5) bytes per vector), and the exact distances of the few survivors read their rows from host memory over PCIe. |
| `n_lists` | `int` | Number of k-means lists (partitions) of the vectors; 0 keeps the vectors in insertion order. With lists, the vectors are stored in list order, and a search first screens the lists nearest to every query, whose exact distances tighten its threshold. The lists change only the speed; the ids stay the insertion order. |
| `kmeans_sample` | `int64_t` | Number of vectors that the k-means centers are trained on. |
| `local_projections` | `bool` | Give every list (n_lists &gt; 0) its own center and principal directions, computed from its vectors, so that the lower bounds of its rows follow the local shape of the collection. The bound holds for every orthonormal projection, so that local projections change only the speed. Vectors added later use the global projection. |
| `int8_rows` | `bool` | Screening rows of int8 codes: every row holds its projection coordinates as int8 codes with one scale per row and its lifted norm and residual norm in fp32 (m rounded up to 64 bytes plus 20 bytes per vector, against 2 (m + 5) bytes in half precision), and the tensor cores accumulate the code products exactly in int32. The quantization error enters the bound of every pair, so that int8 rows return the same exact neighbors with somewhat more survivors. Not combined with lists or label partitions. |

## CertEx search parameters

<a id="neighbors-certex-search-params"></a>
### neighbors::certex::search_params

Parameters of a CertEx search.

```cpp
struct search_params : cuvs::neighbors::search_params {
  int anchors;
  int batch_size;
  int probe_lists;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `anchors` | `int` | Anchors per query whose exact distances set the screening threshold. |
| `batch_size` | `int` | Queries per batch; one certified pass over the screening rows serves a whole batch. |
| `probe_lists` | `int` | Lists nearest to a query (an index with lists) that are screened first, so that the k-th exact distance among their survivors tightens the threshold; 0 skips this probe. |

## CertEx index

<a id="neighbors-certex-index"></a>
### neighbors::certex::index

Exact k-nearest-neighbor index by certified exclusion.

Every vector is stored as a lifted projection onto m principal directions plus the norm of the residual, rounded to half precision, with four certificate coordinates that fold the rounding error of the tensor-core product into the product itself. One pass over these screening rows proves that almost every vector cannot be a neighbor of a query; the survivors receive exact distances in increasing order of their certified lower bounds, until no remaining survivor can enter the top k. Every answer therefore equals exhaustive search in double precision.

The index owns a copy of the vectors as given, the screening rows, the rows of the anchors, and the projection.

```cpp
template <typename T>
struct index : cuvs::neighbors::index {
  double alpha;
  double max_a;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `alpha` | `double` | Common scale of the lifted coordinates, so that every lifted row has norm at most 64. |
| `max_a` | `double` |  |

<a id="neighbors-certex-index-index"></a>
### neighbors::certex::index::index

An empty index, to be filled by deserialize().

```cpp
explicit index(raft::resources const& res);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |

**Returns**

`explicit`

**Additional overload:** `neighbors::certex::index::index`

Construct an empty index for the given metric and width.

```cpp
index(raft::resources const& res,
cuvs::distance::DistanceType metric,
int64_t dim,
int width,
int anchor_stride,
bool host_vectors = false);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `metric` |  | [`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype) |  |
| `dim` |  | `int64_t` |  |
| `width` |  | `int` |  |
| `anchor_stride` |  | `int` |  |
| `host_vectors` |  | `bool` | Default: `false`. |

**Returns**

`void`

<a id="neighbors-certex-index-size"></a>
### neighbors::certex::index::size

Number of vectors.

```cpp
[[nodiscard]] auto size() const noexcept -> int64_t;
```

**Returns**

`int64_t`

<a id="neighbors-certex-index-dim"></a>
### neighbors::certex::index::dim

Dimensionality of the vectors.

```cpp
[[nodiscard]] auto dim() const noexcept -> int64_t;
```

**Returns**

`int64_t`

<a id="neighbors-certex-index-width"></a>
### neighbors::certex::index::width

Number m of principal directions of the screening rows.

```cpp
[[nodiscard]] auto width() const noexcept -> int;
```

**Returns**

`int`

<a id="neighbors-certex-index-row-length"></a>
### neighbors::certex::index::row_length

Halves per screening row: m + 1 lifted and 4 certificate coordinates, padded to 64 or 128n.

```cpp
[[nodiscard]] auto row_length() const noexcept -> int;
```

**Returns**

`int`

<a id="neighbors-certex-index-anchor-stride"></a>
### neighbors::certex::index::anchor_stride

Every anchor_stride()-th vector is an anchor.

```cpp
[[nodiscard]] auto anchor_stride() const noexcept -> int;
```

**Returns**

`int`

<a id="neighbors-certex-index-metric"></a>
### neighbors::certex::index::metric

Distance metric of the index.

```cpp
[[nodiscard]] auto metric() const noexcept -> cuvs::distance::DistanceType;
```

**Returns**

[`cuvs::distance::DistanceType`](/api-reference/cpp-api-distance-distance#distance-distancetype)

<a id="neighbors-certex-index-screening-bytes-per-vector"></a>
### neighbors::certex::index::screening_bytes_per_vector

Bytes of device memory per vector taken by the screening rows and the anchor rows.

```cpp
[[nodiscard]] auto screening_bytes_per_vector() const noexcept -> double;
```

**Returns**

`double`

<a id="neighbors-certex-index-host-vectors"></a>
### neighbors::certex::index::host_vectors

Whether the full-precision vectors live in pinned host memory.

```cpp
[[nodiscard]] auto host_vectors() const noexcept -> bool;
```

**Returns**

`bool`

<a id="neighbors-certex-index-vectors-ptr"></a>
### neighbors::certex::index::vectors_ptr

The vectors used for the exact distances, [capacity, dim] row major (the first size() rows are

```cpp
auto vectors_ptr() noexcept -> T*;
```

valid), in device memory or in pinned host memory that the device reads directly.

**Returns**

`T*`

<a id="neighbors-certex-index-anchor-vectors"></a>
### neighbors::certex::index::anchor_vectors

Full-precision vectors of the anchors 0, s, 2s, ..., always in device memory.

```cpp
auto anchor_vectors() noexcept -> raft::device_matrix_view<T, int64_t, raft::row_major>;
```

**Returns**

`raft::device_matrix_view<T, int64_t, raft::row_major>`

<a id="neighbors-certex-index-rows"></a>
### neighbors::certex::index::rows

Half-precision screening rows with their certificates, [capacity, row_length].

```cpp
auto rows() noexcept -> raft::device_matrix_view<half, int64_t, raft::row_major>;
```

**Returns**

`raft::device_matrix_view<half, int64_t, raft::row_major>`

<a id="neighbors-certex-index-anchor-rows"></a>
### neighbors::certex::index::anchor_rows

Screening rows of the anchors 0, s, 2s, ..., [anchor capacity, row_length].

```cpp
auto anchor_rows() noexcept -> raft::device_matrix_view<half, int64_t, raft::row_major>;
```

**Returns**

`raft::device_matrix_view<half, int64_t, raft::row_major>`

<a id="neighbors-certex-index-n-anchors"></a>
### neighbors::certex::index::n_anchors

Number of anchors, ceil(size() / anchor_stride()).

```cpp
[[nodiscard]] auto n_anchors() const noexcept -> int64_t;
```

**Returns**

`int64_t`

<a id="neighbors-certex-index-center"></a>
### neighbors::certex::index::center

Center of the projection, [dim] (the origin for inner products).

```cpp
auto center() noexcept -> raft::device_vector_view<float, int64_t>;
```

**Returns**

`raft::device_vector_view<float, int64_t>`

<a id="neighbors-certex-index-basis"></a>
### neighbors::certex::index::basis

Orthonormal principal directions, [width, dim].

```cpp
auto basis() noexcept -> raft::device_matrix_view<float, int64_t, raft::row_major>;
```

**Returns**

`raft::device_matrix_view<float, int64_t, raft::row_major>`

<a id="neighbors-certex-index-removed"></a>
### neighbors::certex::index::removed

Removal flags, [capacity]: 1 marks a removed vector, which no search returns.

```cpp
auto removed() noexcept -> raft::device_vector_view<uint8_t, int64_t>;
```

**Returns**

`raft::device_vector_view<uint8_t, int64_t>`

<a id="neighbors-certex-index-n-removed"></a>
### neighbors::certex::index::n_removed

Number of removed vectors; ids stay stable, so that size() counts them too.

```cpp
[[nodiscard]] auto n_removed() const noexcept -> int64_t;
```

**Returns**

`int64_t`

<a id="neighbors-certex-index-labels"></a>
### neighbors::certex::index::labels

Label of every vector, [capacity], for filtered search by label (-1 = no label).

```cpp
auto labels() noexcept -> raft::device_vector_view<int32_t, int64_t>;
```

**Returns**

`raft::device_vector_view<int32_t, int64_t>`

<a id="neighbors-certex-index-has-labels"></a>
### neighbors::certex::index::has_labels

Whether labels were set, so that a search may restrict a query to its label.

```cpp
[[nodiscard]] auto has_labels() const noexcept -> bool;
```

**Returns**

`bool`

<a id="neighbors-certex-index-certificate-maxima"></a>
### neighbors::certex::index::certificate_maxima

Database maxima of the certificate coordinates, which the search margin uses.

```cpp
struct certificate_maxima {
  double lifted;
} maxima;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `lifted` | `double` |  |

<a id="neighbors-certex-index-sync-state"></a>
### neighbors::certex::index::sync_state

Synchronization of concurrent use: searches hold `rw` shared for every batch, so that the

number of vectors and the certificate maxima that a batch reads stay fixed; updates (extend, remove, set_labels) are serialized by `writer`, write their rows beyond the published vectors, and publish under `rw` exclusive. Searches may therefore run concurrently with updates, each thread with its own raft::resources.

```cpp
struct sync_state {
  std::shared_mutex rw;
  std::mutex writer;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `rw` | `std::shared_mutex` |  |
| `writer` | `std::mutex` |  |

<a id="neighbors-certex-index-capacity"></a>
### neighbors::certex::index::capacity

Number of vectors that the allocated buffers hold.

```cpp
[[nodiscard]] auto capacity() const noexcept -> int64_t;
```

**Returns**

`int64_t`

<a id="neighbors-certex-index-int8-rows"></a>
### neighbors::certex::index::int8_rows

int8 screening rows (index_params::int8_rows): codes(), [capacity, code_length()], and

```cpp
[[nodiscard]] auto int8_rows() const noexcept -> bool;
```

row_scalars(), [capacity, 5], of every vector: a / 2 (0 for inner products), the scale s, the norm of the rounding error \|e\|, the norm of the codes \|c\|, and the residual norm r; max_a() is the largest lifted squared norm a.

**Returns**

`bool`

<a id="neighbors-certex-index-use-int8-rows"></a>
### neighbors::certex::index::use_int8_rows

Switch an empty index to int8 screening rows.

```cpp
void use_int8_rows() noexcept;
```

**Returns**

`void`

<a id="neighbors-certex-index-n-partitioned"></a>
### neighbors::certex::index::n_partitioned

Label partition (partition_by_labels()): a copy of the screening rows of the vectors

```cpp
[[nodiscard]] auto n_partitioned() const noexcept -> int64_t;
```

[0, n_partitioned()) sorted by label, so that the vectors of every label occupy the rows [label_offsets()[l], label_offsets()[l + 1]). partition_ids() maps a row of the copy to the id of its vector. Vectors added later are searched without the partition until the next call.

**Returns**

`int64_t`

<a id="neighbors-certex-index-label-offsets"></a>
### neighbors::certex::index::label_offsets

Host copy of the label offsets, [number of labels + 1].

```cpp
[[nodiscard]] auto label_offsets() const noexcept -> const std::vector<int64_t>&;
```

**Returns**

`const std::vector<int64_t>&`

<a id="neighbors-certex-index-n-lists"></a>
### neighbors::certex::index::n_lists

k-means lists (index_params::n_lists): the rows [list_offsets()[j], list_offsets()[j + 1]) hold

```cpp
[[nodiscard]] auto n_lists() const noexcept -> int;
```

the vectors of list j, which covers the vectors stored when the index was built; later vectors follow them in insertion order. row_ids() maps a row to the id of its vector and id_rows() an id to its row; both are empty without lists, where rows and ids coincide.

**Returns**

`int`

<a id="neighbors-certex-index-mapped"></a>
### neighbors::certex::index::mapped

Whether rows and ids differ (an index with lists).

```cpp
[[nodiscard]] auto mapped() const noexcept -> bool;
```

**Returns**

`bool`

<a id="neighbors-certex-index-set-lists"></a>
### neighbors::certex::index::set_lists

Install the lists of a build: offsets, centers, and the map of rows to ids (inverted here).

```cpp
void set_lists(std::vector<int64_t>&& offsets,
raft::device_matrix<float, int64_t, raft::row_major>&& centers,
raft::device_vector<int32_t, int64_t>&& row_ids,
raft::device_vector<int32_t, int64_t>&& id_rows) noexcept;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `offsets` |  | `std::vector<int64_t>&&` |  |
| `centers` |  | `raft::device_matrix<float, int64_t, raft::row_major>&&` |  |
| `row_ids` |  | `raft::device_vector<int32_t, int64_t>&&` |  |
| `id_rows` |  | `raft::device_vector<int32_t, int64_t>&&` |  |

**Returns**

`void`

<a id="neighbors-certex-index-local-projections"></a>
### neighbors::certex::index::local_projections

Local projections (index_params::local_projections): list j has the center local_centers()[j],

```cpp
[[nodiscard]] auto local_projections() const noexcept -> bool;
```

the directions local_bases()[j * width(), (j + 1) * width()), the scale local_alpha()[j], and the certificate maxima local_maxima()[j] of its rows.

**Returns**

`bool`

<a id="neighbors-certex-index-set-partition"></a>
### neighbors::certex::index::set_partition

Install a label partition (partition_by_labels()), or drop it with n = 0.

```cpp
void set_partition(raft::device_matrix<half, int64_t, raft::row_major>&& rows,
raft::device_vector<int32_t, int64_t>&& ids,
std::vector<int64_t>&& offsets,
int64_t n) noexcept;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `rows` |  | `raft::device_matrix<half, int64_t, raft::row_major>&&` |  |
| `ids` |  | `raft::device_vector<int32_t, int64_t>&&` |  |
| `offsets` |  | `std::vector<int64_t>&&` |  |
| `n` |  | `int64_t` |  |

**Returns**

`void`

<a id="neighbors-certex-index-reserve"></a>
### neighbors::certex::index::reserve

Grow the capacity to at least n vectors, keeping the stored rows.

```cpp
void reserve(raft::resources const& res, int64_t n);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `n` |  | `int64_t` |  |

**Returns**

`void`

<a id="neighbors-certex-index-set-size"></a>
### neighbors::certex::index::set_size

Set the number of stored vectors after rows were appended.

```cpp
void set_size(int64_t n) noexcept;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `n` |  | `int64_t` |  |

**Returns**

`void`

## CertEx index build

<a id="neighbors-certex-build"></a>
### neighbors::certex::build

Build a CertEx index.

```cpp
auto build(raft::resources const& res,
index_params const& params,
raft::device_matrix_view<const float, int64_t, raft::row_major> dataset) -> index<float>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`index_params const&`](/api-reference/cpp-api-neighbors-certex#neighbors-certex-index-params) | index parameters |
| `dataset` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device matrix view [n_rows, dim] |

**Returns**

[`index<float>`](/api-reference/cpp-api-neighbors-certex#neighbors-certex-index)

**Additional overload:** `neighbors::certex::build`

Build a CertEx index from vectors in host memory, which may exceed device memory.

```cpp
auto build(raft::resources const& res,
index_params const& params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset) -> index<float>;
```

The vectors are streamed to the device in chunks; with params.host_vectors, they stay in pinned host memory and only the screening rows and the anchors occupy device memory.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `params` |  | [`index_params const&`](/api-reference/cpp-api-neighbors-certex#neighbors-certex-index-params) |  |
| `dataset` |  | `raft::host_matrix_view<const float, int64_t, raft::row_major>` |  |

**Returns**

[`index<float>`](/api-reference/cpp-api-neighbors-certex#neighbors-certex-index)

## CertEx index extend

<a id="neighbors-certex-extend"></a>
### neighbors::certex::extend

Append vectors with the projection of the index.

```cpp
void extend(raft::resources const& res,
raft::device_matrix_view<const float, int64_t, raft::row_major> new_vectors,
index<float>* idx);
```

The bound holds for every orthonormal projection and every center, and the certificate maxima rise to cover the new rows, so that every search stays exact without a rebuild. The new vectors receive the ids size(), size() + 1, ... of the index before the call.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `new_vectors` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device matrix view [n_rows, dim] |
| `idx` | inout | [`index<float>*`](/api-reference/cpp-api-neighbors-certex#neighbors-certex-index) | the index |

**Returns**

`void`

**Additional overload:** `neighbors::certex::extend`

Append vectors from host memory (see the device overload).

```cpp
void extend(raft::resources const& res,
raft::host_matrix_view<const float, int64_t, raft::row_major> new_vectors,
index<float>* idx);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `new_vectors` |  | `raft::host_matrix_view<const float, int64_t, raft::row_major>` |  |
| `idx` |  | [`index<float>*`](/api-reference/cpp-api-neighbors-certex#neighbors-certex-index) |  |

**Returns**

`void`

## CertEx index labels

<a id="neighbors-certex-set-labels"></a>
### neighbors::certex::set_labels

Set the labels of the vectors first, first + 1, ..., for filtered search by label.

```cpp
void set_labels(raft::resources const& res,
raft::device_vector_view<const int32_t, int64_t> labels,
index<float>* idx,
int64_t first = 0);
```

A search with query labels returns, for every query, the exact k nearest vectors that carry the label of the query (fewer if the label holds fewer vectors). Vectors without a label set hold -1.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `labels` | in | `raft::device_vector_view<const int32_t, int64_t>` | a device vector view of non-negative labels |
| `idx` | inout | [`index<float>*`](/api-reference/cpp-api-neighbors-certex#neighbors-certex-index) | the index |
| `first` | in | `int64_t` | id of the vector that receives labels[0]<br />Default: `0`. |

**Returns**

`void`

<a id="neighbors-certex-partition-by-labels"></a>
### neighbors::certex::partition_by_labels

Sort a copy of the screening rows by label, so that a filtered search by label screens only the rows of the label of every query.

```cpp
void partition_by_labels(raft::resources const& res, index<float>* idx);
```

A search with query labels then pairs every query with the rows of its label alone, so that its work shrinks with the selectivity; without the partition, it screens every row and keeps only rows of the query's label. The copy takes another 2 row_length() bytes and 4 bytes per vector. Vectors added later are searched without the partition until the next call, and setting the labels of partitioned vectors drops it. Both kinds of search return the same exact neighbors.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `idx` | inout | [`index<float>*`](/api-reference/cpp-api-neighbors-certex#neighbors-certex-index) | an index with labels |

**Returns**

`void`

## CertEx index remove

<a id="neighbors-certex-remove"></a>
### neighbors::certex::remove

Remove vectors by id; no later search returns them.

```cpp
void remove(raft::resources const& res,
raft::device_vector_view<const int64_t, int64_t> ids,
index<float>* idx);
```

A removed row meets the constant coordinate of every query row with the most negative half-precision value, so that it never survives screening; a removed anchor no longer sets a threshold. Ids stay stable and are never reused; removing an id twice has no further effect.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `ids` | in | `raft::device_vector_view<const int64_t, int64_t>` | a device vector view of ids in [0, size()) |
| `idx` | inout | [`index<float>*`](/api-reference/cpp-api-neighbors-certex#neighbors-certex-index) | the index |

**Returns**

`void`

## CertEx index search

<a id="neighbors-certex-search"></a>
### neighbors::certex::search

Filtered search by label: the exact k nearest vectors of the label of every query.

```cpp
void search(raft::resources const& res,
search_params const& params,
index<float> const& idx,
raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
raft::device_vector_view<const int32_t, int64_t> query_labels,
raft::device_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
raft::device_matrix_view<float, int64_t, raft::row_major> distances);
```

Requires labels (set_labels). Entries beyond the number of vectors of a label hold the id -1.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` | in | `raft::resources const&` | raft resources |
| `params` | in | [`search_params const&`](/api-reference/cpp-api-neighbors-certex#neighbors-certex-search-params) | search parameters |
| `idx` | in | [`index<float> const&`](/api-reference/cpp-api-neighbors-certex#neighbors-certex-index) | the index |
| `queries` | in | `raft::device_matrix_view<const float, int64_t, raft::row_major>` | a device matrix view [n_queries, dim] |
| `query_labels` | in | `raft::device_vector_view<const int32_t, int64_t>` | a device vector view [n_queries], one label per query |
| `neighbors` | out | `raft::device_matrix_view<int64_t, int64_t, raft::row_major>` | a device matrix view [n_queries, k], k &lt;= 1024 |
| `distances` | out | `raft::device_matrix_view<float, int64_t, raft::row_major>` | a device matrix view [n_queries, k] |

**Returns**

`void`

## CertEx index serialize

<a id="neighbors-certex-serialize"></a>
### neighbors::certex::serialize

Save the index, including its vectors, so that deserialize() restores it without a rebuild.

```cpp
void serialize(raft::resources const& res, std::ostream& os, index<float> const& idx);
```

The serialization format may change between versions of the library. A file starts with the NumPy descriptor of the vector type, padded to 4 bytes (as for CAGRA), so that a reader can pick the index type before deserializing.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |
| `os` |  | `std::ostream&` |  |
| `idx` |  | [`index<float> const&`](/api-reference/cpp-api-neighbors-certex#neighbors-certex-index) |  |

**Returns**

`void`
