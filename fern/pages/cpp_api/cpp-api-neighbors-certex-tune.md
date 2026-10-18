---
slug: api-reference/cpp-api-neighbors-certex-tune
---

# Certex Tune

_Source header: `cuvs/neighbors/certex_tune.hpp`_

## CertEx configuration tuning

<a id="neighbors-certex-cost-model"></a>
### neighbors::certex::cost_model

Cost model of a search on the current GPU. Every configuration returns the same exact neighbors, so that the model chooses only the speed. The time of a batch of B queries over N rows of L halves with anchor stride s and S survivors and F verified rows per query is

t = a_amac B (N / s) L + a_apair B (N / s)        (anchor stage) + a_mac B N L + a_pair B N + a_surv B S          (certified pass) + a_sort B S + a_fetch B F                       (ordering and verification)

The pass and anchor constants come from microbenchmarks of the kernels on random rows, and the verification constants from the searches on the prefixes of the dataset.

```cpp
struct cost_model {
  double a_mac_large;
  double a_batch;
  double a_mac;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `a_mac_large` | `double` | a_mac and a_pair of the certified pass over large_rows rows or more, where the pass of short rows switches kernels (0: as below). |
| `a_batch` | `double` | Seconds per batch beyond the terms above (launches and synchronizations of a batch). |
| `a_mac` | `double` |  |

<a id="neighbors-certex-tune-params"></a>
### neighbors::certex::tune_params

Parameters of the tuner.

```cpp
struct tune_params {
  std::vector<int> widths;
  std::vector<int> strides;
  int64_t small_prefix;
  int64_t n_queries;
  int batch_size;
  double memory_budget;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `widths` | `std::vector<int>` | Candidate widths m (0: 59, 123, and 251, below the dimension). |
| `strides` | `std::vector<int>` | Candidate anchor strides. |
| `small_prefix` | `int64_t` | Sizes of the two prefixes on which survivors and verified rows are measured. |
| `n_queries` | `int64_t` | Queries of the prefix searches. |
| `batch_size` | `int` | Queries per batch of the predicted search. |
| `memory_budget` | `double` | Device memory available to the index in bytes (0: the free memory of the device). |

<a id="neighbors-certex-tune-candidate"></a>
### neighbors::certex::tune_candidate

One configuration with its prediction.

```cpp
struct tune_candidate {
  int width, anchor_stride;
  double survivors, verified;
  double predicted_qps;
  double device_bytes;
  bool fits;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `anchor_stride` | `int width,` |  |
| `verified` | `double survivors,` |  |
| `predicted_qps` | `double` |  |
| `device_bytes` | `double` |  |
| `fits` | `bool` |  |

<a id="neighbors-certex-tune-result"></a>
### neighbors::certex::tune_result

The tuner's choice and every candidate.

```cpp
struct tune_result {
  int width;
  double predicted_qps;
  cost_model model;
  std::vector<tune_candidate> candidates;
};
```

**Fields**

| Name | Type | Description |
| --- | --- | --- |
| `width` | `int` |  |
| `predicted_qps` | `double` |  |
| `model` | [`cost_model`](/api-reference/cpp-api-neighbors-certex-tune#neighbors-certex-cost-model) |  |
| `candidates` | [`std::vector<tune_candidate>`](/api-reference/cpp-api-neighbors-certex-tune#neighbors-certex-tune-candidate) |  |

<a id="neighbors-certex-calibrate"></a>
### neighbors::certex::calibrate

Calibrate the pass and anchor constants of the cost model on the current GPU.

```cpp
auto calibrate(raft::resources const& res) -> cost_model;
```

**Parameters**

| Name | Direction | Type | Description |
| --- | --- | --- | --- |
| `res` |  | `raft::resources const&` |  |

**Returns**

[`cost_model`](/api-reference/cpp-api-neighbors-certex-tune#neighbors-certex-cost-model)
