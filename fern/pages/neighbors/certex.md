---
slug: user-guide/api-guides/indexing-guide/certex
---

# CertEx

CertEx is an exact nearest-neighbor index. Every search returns the exact `k` nearest stored vectors of every query, the same result as brute-force, while computing full-precision distances for only a small fraction of the dataset.

CertEx works well when results must be exact and brute-force is too slow: ground truth for large collections, deduplication, retrieval where a missed neighbor is not acceptable, and filtered search with exact results.

## Example API Usage

[C API](/api-reference/c-api-neighbors-certex) | [C++ API](/api-reference/cpp-api-neighbors-certex) | [Python API](/api-reference/python-api-neighbors-certex)

### Building an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/certex.h>

cuvsResources_t res;
cuvsCertexIndexParams_t index_params;
cuvsCertexIndex_t index;
DLManagedTensor *dataset;

load_dataset(dataset);

cuvsResourcesCreate(&res);
cuvsCertexIndexParamsCreate(&index_params);
cuvsCertexIndexCreate(&index);

cuvsCertexBuild(res, index_params, dataset, index);

cuvsCertexIndexParamsDestroy(index_params);
cuvsCertexIndexDestroy(index);
cuvsResourcesDestroy(res);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/certex.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
auto dataset = load_dataset();
certex::index_params index_params;

index_params.metric = cuvs::distance::DistanceType::L2Expanded;

auto index = certex::build(res, index_params, dataset);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import certex

dataset = load_data()

index = certex.build(certex.IndexParams(metric="sqeuclidean"), dataset)
```

</Tab>
</Tabs>

### Searching an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/certex.h>

cuvsResources_t res;
cuvsCertexSearchParams_t search_params;
cuvsCertexIndex_t index;
DLManagedTensor *queries;
DLManagedTensor *neighbors;
DLManagedTensor *distances;

// ... build or load index ...
load_queries(queries);
allocate_outputs(neighbors, distances);

cuvsCertexSearchParamsCreate(&search_params);

cuvsFilter no_filter;
no_filter.type = NO_FILTER;
no_filter.addr = 0;

cuvsCertexSearch(res, search_params, index, queries, neighbors, distances, no_filter);

cuvsCertexSearchParamsDestroy(search_params);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/certex.hpp>

using namespace cuvs::neighbors;

raft::device_resources res;
certex::search_params search_params;
auto queries = load_queries(n_queries, dim);

// ... build or load index ...

auto neighbors = raft::make_device_matrix<int64_t, int64_t>(res, n_queries, k);
auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);

certex::search(res, search_params, index, queries, neighbors.view(), distances.view());
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import certex

queries = load_queries()

# ... build or load index ...

distances, neighbors = certex.search(certex.SearchParams(), index, queries, k=10)
```

</Tab>
</Tabs>

### Adding and removing vectors

A CertEx index accepts new vectors and removals without a rebuild. New vectors receive the next consecutive ids; removed vectors are never returned again. Searches may run concurrently with `extend` from other threads.

<Tabs>
<Tab title="C">

```c
cuvsCertexExtend(res, new_vectors, index);
cuvsCertexRemove(res, ids_to_remove, index);
```

</Tab>
<Tab title="C++">

```cpp
certex::extend(res, new_vectors, &index);
certex::remove(res, ids_to_remove, &index);
```

</Tab>
<Tab title="Python">

```python
certex.extend(index, new_vectors)
certex.remove(index, ids_to_remove)
```

</Tab>
</Tabs>

### Saving and loading an index

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/certex.h>

cuvsCertexSerialize(res, "certex_index.bin", index);

cuvsCertexIndex_t loaded;
cuvsCertexIndexCreate(&loaded);
cuvsCertexDeserialize(res, "certex_index.bin", loaded);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/neighbors/certex.hpp>

using namespace cuvs::neighbors;

certex::serialize(res, "certex_index.bin", index);

certex::index<float> loaded(res);
certex::deserialize(res, "certex_index.bin", &loaded);
```

</Tab>
<Tab title="Python">

```python
from cuvs.neighbors import certex

certex.save("certex_index.bin", index)
loaded = certex.load("certex_index.bin")
```

</Tab>
</Tabs>

## How CertEx works

CertEx stores, next to the vectors, one short half-precision screening row per vector: the vector's coordinates along the leading `m` principal directions of the dataset, plus the norm of its remainder. From two screening rows, a tensor-core product gives a lower bound on the exact distance between a query and a vector. The bound accounts for the dropped directions and for every rounding error of the half-precision arithmetic, so it never exceeds the exact distance.

A search runs in three steps for a batch of queries:

1. **Threshold.** The exact distances of each query to a few anchor vectors give an upper bound on its `k`-th nearest distance.
2. **Screening.** One pass over the screening rows keeps only the vectors whose lower bound does not exceed the threshold. Every other vector is provably farther than the `k`-th neighbor.
3. **Verification.** Exact distances of the surviving vectors, in order of their bounds, give the exact `k` nearest neighbors.

Because the exclusion is certified rather than estimated, every setting of the parameters returns the same exact neighbors. The parameters change only the speed and the memory.

On GPUs of compute capability 9.0 (Hopper), screening uses TMA and WGMMA kernels; other GPUs of compute capability 8.0 or later use `mma.sync` kernels. CertEx needs a GPU of compute capability 8.0 or later.

## When to use CertEx

Use CertEx when exact nearest neighbors are required and the dataset is large enough that brute-force search is slow.

Use CertEx for exact ground truth and all-neighbors workloads over large collections.

Use CertEx when vectors must stay exact but device memory is short: with `host_vectors`, the full-precision vectors stay in pinned host memory and device memory holds only the screening rows.

Use an approximate index such as CAGRA or IVF-PQ when approximate results are acceptable and the highest throughput matters more than exactness.

## Using Filters

CertEx supports filtered search with bitset filters. A vector that fails the filter is never returned, and the result is the exact `k` nearest neighbors among the allowed vectors.

<Tabs>
<Tab title="C">

```c
#include <cuvs/neighbors/certex.h>

// A device uint32 bitset with one bit per vector; bit 1 means the vector is allowed.
DLManagedTensor *bitset = make_device_bitset(allowed_vectors, n_vectors);

cuvsFilter prefilter;
prefilter.type = BITSET;
prefilter.addr = (uintptr_t)bitset;

cuvsCertexSearch(res, search_params, index, queries, neighbors, distances, prefilter);
```

</Tab>
<Tab title="C++">

```cpp
#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/certex.hpp>

using namespace cuvs::neighbors;

// A bitset with one bit per vector; clear the bits of the vectors to exclude.
cuvs::core::bitset<uint32_t, int64_t> allowed(res, excluded_ids.view(), n_vectors);
auto bitset_filter = filtering::bitset_filter(allowed.view());

certex::search(
    res, search_params, index, queries, neighbors.view(), distances.view(), bitset_filter);
```

</Tab>
<Tab title="Python">

```python
import cupy as cp
from cuvs.neighbors import certex, filters

# One bit per vector; bit 1 means the vector is allowed.
prefilter = filters.from_bitset(cp.asarray(bitset_host))

distances, neighbors = certex.search(
    certex.SearchParams(), index, queries, k=10, filter=prefilter
)
```

</Tab>
</Tabs>

## Configuration parameters

### Build parameters

| Name | Default | Description |
| --- | --- | --- |
| `metric` | `L2Expanded` / `sqeuclidean` | Distance metric: squared Euclidean, inner product, or cosine. |
| `width` | `0` | Width `m` of the screening rows. `0` picks the smallest width that captures `variance` of the spectrum. |
| `variance` | `0.95` | Fraction of the variance that an automatic width captures. |
| `anchor_stride` | `16` | Every `anchor_stride`-th vector is an anchor of the threshold step. |
| `pca_sample` | `1000000` | Number of vectors the principal directions are computed from. |
| `host_vectors` | `false` | Keep the full-precision vectors in pinned host memory; verification reads the few survivors over PCIe. |
| `n_lists` | `0` | Number of k-means lists. With lists, a search first screens the lists nearest to each query, which tightens its threshold. |
| `local_projections` | `false` | Give every list its own principal directions (requires `n_lists > 0`). |
| `int8_rows` | `false` | Store the screening rows as int8 codes with exact int32 accumulation (not combined with lists). |

### Search parameters

| Name | Default | Description |
| --- | --- | --- |
| `anchors` | `128` | Anchors per query whose exact distances set the screening threshold. |
| `batch_size` | `2048` | Queries per batch; one screening pass serves a whole batch. |
| `probe_lists` | `2` | Lists nearest to each query that are screened first (an index with lists); `0` skips this step. |

## Tuning Considerations

All settings return the same exact neighbors, so tuning trades only speed against memory.

A wider screening row (`width`) gives tighter bounds and fewer survivors to verify, at the cost of more memory and a longer screening pass. The automatic width is a good start; for high-dimensional embeddings, widths between 64 and 256 are typical.

Larger query batches amortize the screening pass over more queries. Increase `batch_size` for throughput and decrease it for latency.

Lists (`n_lists`) help most for clustered data, where the nearest lists give a tight threshold early.

Exact search returns the true distances, but ties at the `k`-th distance can make the order or the identity of tied neighbors differ from another exact method. Compare results by distance rather than by id when ties are possible.

## Memory footprint

Variables:

- `N`: Number of database vectors.
- `D`: Vector dimension.
- `B`: Bytes per dataset value, such as `4` for fp32.
- `m`: Width of the screening rows.
- `Q`: Query batch size.
- `K`: Number of neighbors returned per query.

### Index footprint

The screening rows use about:

$$
\text{rows\_size} \approx N \cdot 2\,(m + 5)
$$

The full-precision vectors use:

$$
\text{dataset\_size} = N \cdot D \cdot B
$$

in device memory, or in pinned host memory with `host_vectors`. The anchors and the principal directions add a small fixed amount.

### Search peak memory usage

Search adds the query batch, the survivors of the screening pass, and the output buffers:

$$
\begin{aligned}
\text{search\_peak}
  \approx&\ \text{rows\_size}
   + \text{dataset\_size} \\
  &+ Q \cdot D \cdot B
   + Q \cdot K \cdot 12
   + \text{survivors}
   + \text{scratch}
\end{aligned}
$$

The survivor buffer grows with the batch size and with the looseness of the bounds. When it does not fit, CertEx splits the batch and continues.
