#ifndef _OUR_SPMM_CUH
#define _OUR_SPMM_CUH
#include "general.cuh"
void our_spmm_balanced(CSR *A_csr, half *B, float *C, const int M, const int N, const int K, int n_iter, double *);
#endif