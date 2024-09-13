#ifndef _SPMM_CUSPARSE_CUH
#define _SPMM_CUSPARSE_CUH
#include <cusparse.h>
#include <cusparseLt.h>
#include "general.cuh"
#define CHECK_CUSPARSE(x)                                                                   \
    if (x != CUSPARSE_STATUS_SUCCESS)                                                       \
    {                                                                                       \
        printf("Cusparse Error %s %s:%d\n", cusparseGetErrorString(x), __FILE__, __LINE__); \
        exit(EXIT_FAILURE);                                                                 \
    }

void perform_cuSPARSELT_24sparsity(half *A, half *B, float *C, int M, int N, int K, int num_iter, double *);
void perform_cuSPARSE_BlockSpMM(half *A, half *B, float *C, int M, int N, int K, int num_iter, int ell_block_size, double *);
void perform_cuSPARSE_SpMM(half *A, half *B, float *C, int M, int N, int K, int num_iter, double *);
#endif