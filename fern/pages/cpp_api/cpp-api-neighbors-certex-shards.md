---
slug: api-reference/cpp-api-neighbors-certex-shards
---

# Certex Shards

_Source header: `cuvs/neighbors/certex_shards.hpp`_

## CertEx over several GPUs

<a id="neighbors-certex-sharded-index"></a>
### neighbors::certex::sharded_index

A CertEx index split into contiguous shards, one CertEx index per GPU.

A search runs in two rounds. First, every shard computes the threshold U of every query from its own anchors; U bounds the k-th distance over the whole collection, because any k distinct vectors bound it, so that the smallest U over the shards does too. Second, every shard searches under this shared threshold and returns the exact k nearest neighbors of its own vectors within it, and the host merges the shard answers by exact distance. Only k pairs of id and distance per shard and query cross the interconnect.

```cpp
template <typename T>
struct sharded_index {
  std::vector<int> devices;
  std::vector<std::unique_ptr<raft::device_resources>> resources;
  std::vector<std::unique_ptr<index<T>>> shards;
  std::vector<int64_t> first_id;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `devices` | `std::vector<int>` | Device of every shard. |
| `resources` | `std::vector<std::unique_ptr<raft::device_resources>>` | Resources of every shard, created on its device. |
| `shards` | `std::vector<std::unique_ptr<index<T>>>` | The index of every shard. |
| `first_id` | `std::vector<int64_t>` | Id of the first vector of every shard; shard s holds the ids first_id[s], first_id[s] + 1, ... |

<a id="neighbors-certex-sharded-index-size"></a>
### neighbors::certex::sharded_index::size

Number of vectors over all shards.

```cpp
[[nodiscard]] auto size() const -> int64_t;
```

**Returns**

`int64_t`

<a id="neighbors-certex-build-sharded"></a>
### neighbors::certex::build_sharded

Build a sharded index: the vectors are split into one contiguous shard per device, and every shard is built on its device in parallel. A device may appear more than once.

```cpp
auto build_sharded(std::vector<int> const& devices,
index_params const& params,
raft::host_matrix_view<const float, int64_t, raft::row_major> dataset)
-> sharded_index<float>;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `devices` | in | `std::vector<int> const&` | CUDA device of every shard |
| `params` | in | `index_params const&` | index parameters of every shard |
| `dataset` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host matrix view [n_rows, dim] |

**Returns**

[`sharded_index<float>`](/api-reference/cpp-api-neighbors-certex-shards#neighbors-certex-sharded-index)

<a id="neighbors-certex-search"></a>
### neighbors::certex::search

Search the exact k nearest neighbors of every query over all shards, with k = neighbors.extent(1); every shard must hold at least k vectors.

```cpp
void search(search_params const& params,
sharded_index<float> const& idx,
raft::host_matrix_view<const float, int64_t, raft::row_major> queries,
raft::host_matrix_view<int64_t, int64_t, raft::row_major> neighbors,
raft::host_matrix_view<float, int64_t, raft::row_major> distances);
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `params` | in | `search_params const&` | search parameters of every shard |
| `idx` | in | [`sharded_index<float> const&`](/api-reference/cpp-api-neighbors-certex-shards#neighbors-certex-sharded-index) | the sharded index |
| `queries` | in | `raft::host_matrix_view<const float, int64_t, raft::row_major>` | a host matrix view [n_queries, dim] |
| `neighbors` | out | `raft::host_matrix_view<int64_t, int64_t, raft::row_major>` | a host matrix view [n_queries, k] of ids |
| `distances` | out | `raft::host_matrix_view<float, int64_t, raft::row_major>` | a host matrix view [n_queries, k], as for a single index |

**Returns**

`void`
