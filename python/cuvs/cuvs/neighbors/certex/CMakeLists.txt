# =============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
# =============================================================================

set(cython_sources certex.pyx)
set(linked_libraries cuvs::cuvs cuvs::c_api)

rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" MODULE_PREFIX neighbors_certex_
)
