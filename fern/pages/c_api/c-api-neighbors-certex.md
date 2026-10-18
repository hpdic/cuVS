---
slug: api-reference/c-api-neighbors-certex
---

# Certex

_Source header: `cuvs/neighbors/certex.h`_

## CertEx C-API index build parameters

<a id="cuvscertexindexparams"></a>
### cuvsCertexIndexParams

Parameters of a CertEx index (cuvs::neighbors::certex::index_params). Every setting returns the same exact neighbors; the settings change only the speed and the device memory.

```c
struct cuvsCertexIndexParams {
  cuvsDistanceType metric;
  int32_t width;
  double variance;
  int32_t anchor_stride;
  int64_t pca_sample;
  bool host_vectors;
  int32_t n_lists;
  bool local_projections;
  bool int8_rows;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `metric` | [`cuvsDistanceType`](/api-reference/c-api-distance-distance#cuvsdistancetype) | L2Expanded, InnerProduct, or CosineExpanded |
| `width` | `int32_t` | Projection width m of the screening rows (0: the smallest width that explains `variance`) |
| `variance` | `double` | Fraction of the variance that the automatic width explains |
| `anchor_stride` | `int32_t` | Every anchor_stride-th vector is an anchor of the threshold stage |
| `pca_sample` | `int64_t` | Vectors sampled for the principal directions |
| `host_vectors` | `bool` | Keep the full-precision vectors in pinned host memory |
| `n_lists` | `int32_t` | k-means lists (0: none); a search first screens the nearest lists of every query |
| `local_projections` | `bool` | Every list gets its own projection (requires n_lists &gt; 0) |
| `int8_rows` | `bool` | int8 screening rows |

<a id="cuvscertexindexparamscreate"></a>
### cuvsCertexIndexParamsCreate

Allocate index parameters with the defaults of the C++ API.

```c
cuvsError_t cuvsCertexIndexParamsCreate(cuvsCertexIndexParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` |  | [`cuvsCertexIndexParams_t*`](/api-reference/c-api-neighbors-certex#cuvscertexindexparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscertexindexparamsdestroy"></a>
### cuvsCertexIndexParamsDestroy

Free index parameters.

```c
cuvsError_t cuvsCertexIndexParamsDestroy(cuvsCertexIndexParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` |  | [`cuvsCertexIndexParams_t`](/api-reference/c-api-neighbors-certex#cuvscertexindexparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## CertEx C-API search parameters

<a id="cuvscertexsearchparams"></a>
### cuvsCertexSearchParams

Parameters of a CertEx search (cuvs::neighbors::certex::search_params).

```c
struct cuvsCertexSearchParams {
  int32_t anchors;
  int32_t batch_size;
  int32_t probe_lists;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `anchors` | `int32_t` | Anchors per query whose exact distances set the threshold |
| `batch_size` | `int32_t` | Queries per batch |
| `probe_lists` | `int32_t` | Nearest lists that an index with lists screens first |

<a id="cuvscertexsearchparamscreate"></a>
### cuvsCertexSearchParamsCreate

Allocate search parameters with the defaults of the C++ API.

```c
cuvsError_t cuvsCertexSearchParamsCreate(cuvsCertexSearchParams_t* params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` |  | [`cuvsCertexSearchParams_t*`](/api-reference/c-api-neighbors-certex#cuvscertexsearchparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscertexsearchparamsdestroy"></a>
### cuvsCertexSearchParamsDestroy

Free search parameters.

```c
cuvsError_t cuvsCertexSearchParamsDestroy(cuvsCertexSearchParams_t params);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` |  | [`cuvsCertexSearchParams_t`](/api-reference/c-api-neighbors-certex#cuvscertexsearchparams) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## CertEx C-API index

<a id="cuvscertexindex"></a>
### cuvsCertexIndex

An opaque CertEx index of float32, float16, int8, or uint8 vectors.

```c
typedef struct {
  uintptr_t addr;
  DLDataType dtype;
} cuvsCertexIndex;
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `addr` | `uintptr_t` |  |
| `dtype` | `DLDataType` |  |

<a id="cuvscertexindexcreate"></a>
### cuvsCertexIndexCreate

Allocate an empty index handle.

```c
cuvsError_t cuvsCertexIndexCreate(cuvsCertexIndex_t* index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | [`cuvsCertexIndex_t*`](/api-reference/c-api-neighbors-certex#cuvscertexindex) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscertexindexdestroy"></a>
### cuvsCertexIndexDestroy

Free an index handle and its index.

```c
cuvsError_t cuvsCertexIndexDestroy(cuvsCertexIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | [`cuvsCertexIndex_t`](/api-reference/c-api-neighbors-certex#cuvscertexindex) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscertexindexgetsize"></a>
### cuvsCertexIndexGetSize

Number of vectors added to the index (removed ones included).

```c
cuvsError_t cuvsCertexIndexGetSize(cuvsCertexIndex_t index, int64_t* size);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | [`cuvsCertexIndex_t`](/api-reference/c-api-neighbors-certex#cuvscertexindex) |  |
| `size` |  | `int64_t*` |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscertexindexgetdim"></a>
### cuvsCertexIndexGetDim

Dimension of the vectors of the index.

```c
cuvsError_t cuvsCertexIndexGetDim(cuvsCertexIndex_t index, int64_t* dim);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `index` |  | [`cuvsCertexIndex_t`](/api-reference/c-api-neighbors-certex#cuvscertexindex) |  |
| `dim` |  | `int64_t*` |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## CertEx C-API build, extend, remove, labels

<a id="cuvscertexbuild"></a>
### cuvsCertexBuild

Build an index from a dataset [n_rows, dim] of float32, float16, int8, or uint8 in device or host memory (a host dataset is streamed to the GPU in chunks).

```c
cuvsError_t cuvsCertexBuild(cuvsResources_t res,
cuvsCertexIndexParams_t params,
DLManagedTensor* dataset,
cuvsCertexIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) |  |
| `params` |  | [`cuvsCertexIndexParams_t`](/api-reference/c-api-neighbors-certex#cuvscertexindexparams) |  |
| `dataset` |  | `DLManagedTensor*` |  |
| `index` |  | [`cuvsCertexIndex_t`](/api-reference/c-api-neighbors-certex#cuvscertexindex) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscertexextend"></a>
### cuvsCertexExtend

Append vectors [n, dim] (device or host, the type of the index); they receive the next consecutive ids.

```c
cuvsError_t cuvsCertexExtend(cuvsResources_t res,
DLManagedTensor* new_vectors,
cuvsCertexIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) |  |
| `new_vectors` |  | `DLManagedTensor*` |  |
| `index` |  | [`cuvsCertexIndex_t`](/api-reference/c-api-neighbors-certex#cuvscertexindex) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscertexremove"></a>
### cuvsCertexRemove

Remove vectors by id (an int64 device vector); no later search returns them.

```c
cuvsError_t cuvsCertexRemove(cuvsResources_t res,
DLManagedTensor* ids,
cuvsCertexIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) |  |
| `ids` |  | `DLManagedTensor*` |  |
| `index` |  | [`cuvsCertexIndex_t`](/api-reference/c-api-neighbors-certex#cuvscertexindex) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscertexsetlabels"></a>
### cuvsCertexSetLabels

Set the labels (an int32 device vector) of the vectors first, first + 1, ... for filtered search by label.

```c
cuvsError_t cuvsCertexSetLabels(cuvsResources_t res,
DLManagedTensor* labels,
int64_t first,
cuvsCertexIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) |  |
| `labels` |  | `DLManagedTensor*` |  |
| `first` |  | `int64_t` |  |
| `index` |  | [`cuvsCertexIndex_t`](/api-reference/c-api-neighbors-certex#cuvscertexindex) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscertexpartitionbylabels"></a>
### cuvsCertexPartitionByLabels

Sort a copy of the screening rows by label, so that a search by label screens only the rows of the label of every query (index with the global projection).

```c
cuvsError_t cuvsCertexPartitionByLabels(cuvsResources_t res, cuvsCertexIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) |  |
| `index` |  | [`cuvsCertexIndex_t`](/api-reference/c-api-neighbors-certex#cuvscertexindex) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## CertEx C-API search

<a id="cuvscertexsearch"></a>
### cuvsCertexSearch

Exact k nearest neighbors of every query, k = the second extent of `neighbors`.

```c
cuvsError_t cuvsCertexSearch(cuvsResources_t res,
cuvsCertexSearchParams_t params,
cuvsCertexIndex_t index,
DLManagedTensor* queries,
DLManagedTensor* neighbors,
DLManagedTensor* distances,
cuvsFilter filter);
```

queries [n_queries, dim] (device, the type of the index), neighbors [n_queries, k] int64 and distances [n_queries, k] float32 (device). The distances follow cuVS brute force: squared Euclidean distances, inner products, or cosine distances. `filter` is NO_FILTER, a BITSET over the ids, or a BITMAP of one row of bits per query (uint32 device words); the result then holds the exact k nearest vectors that the filter keeps, with id -1 where fewer exist.

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) |  |
| `params` |  | [`cuvsCertexSearchParams_t`](/api-reference/c-api-neighbors-certex#cuvscertexsearchparams) |  |
| `index` |  | [`cuvsCertexIndex_t`](/api-reference/c-api-neighbors-certex#cuvscertexindex) |  |
| `queries` |  | `DLManagedTensor*` |  |
| `neighbors` |  | `DLManagedTensor*` |  |
| `distances` |  | `DLManagedTensor*` |  |
| `filter` |  | [`cuvsFilter`](/api-reference/c-api-neighbors-common#cuvsfilter) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscertexsearchwithlabels"></a>
### cuvsCertexSearchWithLabels

Exact k nearest neighbors of every query among the vectors of its label (query_labels: an int32 device vector [n_queries]); requires labels (cuvsCertexSetLabels).

```c
cuvsError_t cuvsCertexSearchWithLabels(cuvsResources_t res,
cuvsCertexSearchParams_t params,
cuvsCertexIndex_t index,
DLManagedTensor* queries,
DLManagedTensor* query_labels,
DLManagedTensor* neighbors,
DLManagedTensor* distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) |  |
| `params` |  | [`cuvsCertexSearchParams_t`](/api-reference/c-api-neighbors-certex#cuvscertexsearchparams) |  |
| `index` |  | [`cuvsCertexIndex_t`](/api-reference/c-api-neighbors-certex#cuvscertexindex) |  |
| `queries` |  | `DLManagedTensor*` |  |
| `query_labels` |  | `DLManagedTensor*` |  |
| `neighbors` |  | `DLManagedTensor*` |  |
| `distances` |  | `DLManagedTensor*` |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

## CertEx C-API serialize

<a id="cuvscertexserialize"></a>
### cuvsCertexSerialize

Save the index, including its vectors, to a file.

```c
cuvsError_t cuvsCertexSerialize(cuvsResources_t res,
const char* filename,
cuvsCertexIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) |  |
| `filename` |  | `const char*` |  |
| `index` |  | [`cuvsCertexIndex_t`](/api-reference/c-api-neighbors-certex#cuvscertexindex) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)

<a id="cuvscertexdeserialize"></a>
### cuvsCertexDeserialize

Load an index saved by cuvsCertexSerialize into an empty index handle; the vector type comes from the file.

```c
cuvsError_t cuvsCertexDeserialize(cuvsResources_t res,
const char* filename,
cuvsCertexIndex_t index);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | [`cuvsResources_t`](/api-reference/c-api-core-c-api#cuvsresources-t) |  |
| `filename` |  | `const char*` |  |
| `index` |  | [`cuvsCertexIndex_t`](/api-reference/c-api-neighbors-certex#cuvscertexindex) |  |

**Returns**

[`cuvsError_t`](/api-reference/c-api-core-c-api#cuvserror-t)
