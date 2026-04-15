#include <cufft.h>

extern "C" int CustomRFFT3D(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                         void *extra) {
    if (nparam != 2) return 1;
    cufftReal *input_tensor = static_cast<cufftReal *>(params[0]);
    cufftComplex *output_tensor = static_cast<cufftComplex *>(params[1]);
    int fftx = shapes[0][0];
    int ffty = shapes[0][1];
    int fftz = shapes[0][2];
    // if (strcmp(dtypes[0], "float32") != 0) return 2;
    // if (strcmp(dtypes[1], "complex64") != 0) return 2;
    cufftHandle FFT_plan_r2c;

    cufftPlan3d(&FFT_plan_r2c, fftx, ffty, fftz, CUFFT_R2C);
    cufftExecR2C(FFT_plan_r2c, input_tensor, output_tensor);
    return 0;
}

extern "C" int CustomIRFFT3D(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                         void *extra) {
    if (nparam != 2) return 1;
    cufftComplex *input_tensor = static_cast<cufftComplex *>(params[0]);
    cufftReal *output_tensor = static_cast<cufftReal *>(params[1]);
    int fftx = shapes[0][0];
    int ffty = shapes[0][1];
    int fftz = shapes[0][2];
    cufftHandle FFT_plan_c2r;

    cufftPlan3d(&FFT_plan_c2r, fftx, ffty, (fftz-1)*2, CUFFT_C2R);
    cufftExecC2R(FFT_plan_c2r, reinterpret_cast<cufftComplex *>(input_tensor), output_tensor);

    return 0;
}