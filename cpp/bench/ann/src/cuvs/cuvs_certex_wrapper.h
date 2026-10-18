/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/ann_types.hpp"
#include "cuvs_ann_bench_utils.h"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/certex.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <memory>
#include <string>

namespace cuvs::bench {

/** CertEx: exact k nearest neighbors by certified exclusion (every search returns the exact top k).
 */
template <typename T>
class cuvs_certex : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;

  struct search_param : public search_param_base {
    cuvs::neighbors::certex::search_params certex_params;
  };

  using build_param = cuvs::neighbors::certex::index_params;

  cuvs_certex(Metric metric, int dim, const build_param& param)
    : algo<T>(metric, dim), index_params_(param), dimension_(dim)
  {
    index_params_.metric = parse_metric_type(metric);
  }

  void build(const T* dataset, size_t nrow) final
  {
    // the build streams a host dataset to the GPU in chunks
    index_ = std::make_shared<cuvs::neighbors::certex::index<T>>(cuvs::neighbors::certex::build(
      handle_,
      index_params_,
      raft::make_host_matrix_view<const T, int64_t>(dataset, nrow, dimension_)));
  }

  void set_search_param(const search_param_base& param, const void* filter_bitset) override
  {
    filter_        = make_cuvs_filter(filter_bitset, index_->size());
    search_params_ = dynamic_cast<const search_param&>(param).certex_params;
  }

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override
  {
    static_assert(sizeof(algo_base::index_type) == sizeof(int64_t));
    auto sp       = search_params_;
    sp.batch_size = std::max(sp.batch_size, 1);
    cuvs::neighbors::certex::search(
      handle_,
      sp,
      *index_,
      raft::make_device_matrix_view<const T, int64_t>(queries, batch_size, dimension_),
      raft::make_device_matrix_view<int64_t, int64_t>(
        reinterpret_cast<int64_t*>(neighbors), batch_size, k),
      raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k),
      *filter_);
  }

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return handle_.get_sync_stream().get();
  }

  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHostMmap;
    property.query_memory_type   = MemoryType::kDevice;
    return property;
  }

  void save(const std::string& file) const override
  {
    cuvs::neighbors::certex::serialize(handle_, file, *index_);
  }

  void load(const std::string& file) override
  {
    index_ = std::make_shared<cuvs::neighbors::certex::index<T>>(handle_);
    cuvs::neighbors::certex::deserialize(handle_, file, index_.get());
  }

  std::unique_ptr<algo<T>> copy() override { return std::make_unique<cuvs_certex<T>>(*this); }

 private:
  // handle_ must go first to make sure it dies last and all memory allocated in pool
  configured_raft_resources handle_{};
  build_param index_params_;
  cuvs::neighbors::certex::search_params search_params_;
  std::shared_ptr<cuvs::neighbors::certex::index<T>> index_;
  int dimension_;
  std::shared_ptr<cuvs::neighbors::filtering::base_filter> filter_;
};

}  // namespace cuvs::bench
