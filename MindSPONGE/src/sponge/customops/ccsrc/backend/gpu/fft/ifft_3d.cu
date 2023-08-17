/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cufft.h>
#include "ifft_3d.cuh"

extern "C"
int IFFT3D(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream, void *extra) {
    if (nparam != 2 || ndims[0] != 3) {
        return 1;
    }
    if (strcmp(dtypes[0], "complex64") != 0) {
        return 2;
    }

    cudaStream_t custream = static_cast<cudaStream_t>(stream);
    void *input = params[0];
    void *output = params[1];

    int fftx = shapes[0][0];
    int ffty = shapes[0][1];
    int fftz = shapes[0][2];

    cufftHandle FFT_plan_c2r;
    cufftPlan3d(&FFT_plan_c2r, fftx, ffty, (fftz - 1) * 2, CUFFT_C2R);
    cufftSetStream(FFT_plan_c2r, custream);
    cufftExecC2R(FFT_plan_c2r, static_cast<cufftComplex *>(input), static_cast<cufftReal *>(output));

    return 0;
}
