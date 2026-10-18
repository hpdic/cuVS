#
# SPDX-FileCopyrightText: Copyright (c) 2026, Dongfang Zhao <dzhao@cs.washington.edu>
# SPDX-License-Identifier: Apache-2.0
#
# cython: language_level=3

from libc.stdint cimport int32_t, int64_t, uintptr_t
from libcpp cimport bool

from cuvs.common.c_api cimport cuvsError_t, cuvsResources_t
from cuvs.common.cydlpack cimport DLDataType, DLManagedTensor
from cuvs.distance_type cimport cuvsDistanceType
from cuvs.neighbors.filters.filters cimport cuvsFilter


cdef extern from "cuvs/neighbors/certex.h" nogil:

    ctypedef struct cuvsCertexIndexParams:
        cuvsDistanceType metric
        int32_t width
        double variance
        int32_t anchor_stride
        int64_t pca_sample
        bool host_vectors
        int32_t n_lists
        bool local_projections
        bool int8_rows

    ctypedef cuvsCertexIndexParams* cuvsCertexIndexParams_t

    ctypedef struct cuvsCertexSearchParams:
        int32_t anchors
        int32_t batch_size
        int32_t probe_lists

    ctypedef cuvsCertexSearchParams* cuvsCertexSearchParams_t

    ctypedef struct cuvsCertexIndex:
        uintptr_t addr
        DLDataType dtype

    ctypedef cuvsCertexIndex* cuvsCertexIndex_t

    cuvsError_t cuvsCertexIndexParamsCreate(cuvsCertexIndexParams_t* params)
    cuvsError_t cuvsCertexIndexParamsDestroy(cuvsCertexIndexParams_t params)
    cuvsError_t cuvsCertexSearchParamsCreate(cuvsCertexSearchParams_t* params)
    cuvsError_t cuvsCertexSearchParamsDestroy(cuvsCertexSearchParams_t params)
    cuvsError_t cuvsCertexIndexCreate(cuvsCertexIndex_t* index)
    cuvsError_t cuvsCertexIndexDestroy(cuvsCertexIndex_t index)
    cuvsError_t cuvsCertexIndexGetSize(cuvsCertexIndex_t index, int64_t* size)
    cuvsError_t cuvsCertexIndexGetDim(cuvsCertexIndex_t index, int64_t* dim)

    cuvsError_t cuvsCertexBuild(cuvsResources_t res,
                                cuvsCertexIndexParams_t params,
                                DLManagedTensor* dataset,
                                cuvsCertexIndex_t index) except +

    cuvsError_t cuvsCertexExtend(cuvsResources_t res,
                                 DLManagedTensor* new_vectors,
                                 cuvsCertexIndex_t index) except +

    cuvsError_t cuvsCertexRemove(cuvsResources_t res,
                                 DLManagedTensor* ids,
                                 cuvsCertexIndex_t index) except +

    cuvsError_t cuvsCertexSearch(cuvsResources_t res,
                                 cuvsCertexSearchParams_t params,
                                 cuvsCertexIndex_t index,
                                 DLManagedTensor* queries,
                                 DLManagedTensor* neighbors,
                                 DLManagedTensor* distances,
                                 cuvsFilter filter) except +

    cuvsError_t cuvsCertexSerialize(cuvsResources_t res,
                                    const char* filename,
                                    cuvsCertexIndex_t index) except +

    cuvsError_t cuvsCertexDeserialize(cuvsResources_t res,
                                      const char* filename,
                                      cuvsCertexIndex_t index) except +
