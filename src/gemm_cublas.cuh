#ifndef _GEMM_CUBLAS_CUH
#define _GEMM_CUBLAS_CUH
#include "general.cuh"
#include <cublas_v2.h>
#include <library_types.h>
#define CHECK_CUBLAS(x)                                                                     \
    if ((x) != CUBLAS_STATUS_SUCCESS)                                                       \
    {                                                                                       \
        printf("Cublas Error %s at %s:%d\n", cublasGetStatusString(x), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                                 \
    }
void perform_GemmEx(half *A, half *B, float *C, int M, int N, int K, int n_iter, double *);
#endif