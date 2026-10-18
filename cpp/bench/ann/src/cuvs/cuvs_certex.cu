/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cuvs_certex_wrapper.h"

namespace cuvs::bench {
template class cuvs_certex<float>;
template class cuvs_certex<half>;
template class cuvs_certex<uint8_t>;
template class cuvs_certex<int8_t>;
}  // namespace cuvs::bench
