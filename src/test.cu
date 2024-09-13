#include "arg_parse.h"
#include "file_input.cuh"
#include "gemm_cublas.cuh"
#include "general.cuh"
#include "our_spmm.cuh"
#include "spmm_cusparse.cuh"
#include <algorithm>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/types.h>
std::vector<std::string> split(std::string str, char Delimiter)
{
    std::istringstream iss(str);
    std::string buffer;

    std::vector<std::string> result;

    while (getline(iss, buffer, Delimiter))
    {
        result.push_back(buffer);
    }

    return result;
}
int main(int argc, char *argv[])
{
    half *A, *B;
    float *C_gemmex, *C_comparing; //*C_cusparse, *C_bspmm, *C_our, *C_our_opt_v1, *C_our_opt_v2, *C_our_opt_v12;
    double e_gemmex, e_cusparse, e_our;

    std::ifstream fin;
    int M, N, K;
    int nnz;
    if (argc <= 1)
    {
        puts("Usage : ./test {matrix.smtx}");
        exit(EXIT_FAILURE);
    }
    fin.open(argv[1], std::ifstream::in);

    smtx_input_to_Dense(fin, &A, argv[1], &M, &K, &nnz);
    // return;
    N = 256;

    e_gemmex = e_cusparse = e_our = -1.0;
    for (int i = 0; i < 10; i++)
        warm_up_gpu_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();

    printf("Input file - %s\n", argv[1]);

    prepare_cpu_matrix(&B, K, N, 0.1);
    prepare_cpu_matrix(&C_gemmex, M, N, 1, true);
    prepare_cpu_matrix(&C_comparing, M, N, 1, true);
    // print_mat(A, M, K);
    perform_GemmEx(A, B, C_gemmex, M, N, K, 10, &e_gemmex);

    memset(C_comparing, 0.0f, M * N * sizeof(float));
    perform_cuSPARSE_SpMM(A, B, C_comparing, M, N, K, 10, &e_cusparse);
    if (compare_mat(C_gemmex, C_comparing, M * N, DEFAULT_TOL) == false)
    {
        printf("\033[35mvalidation failed (Cusparse)\033[0m\n");
    }

    CSR *csrA;
    convert_dense_to_csr(A, &csrA, M, K);

    memset(C_comparing, 0.0f, M * N * sizeof(float));
    our_spmm_balanced(csrA, B, C_comparing, M, N, K, 10, &e_our);

    if (compare_mat(C_gemmex, C_comparing, M * N, DEFAULT_TOL) == false)
    {
        printf("\033[35mvalidation failed (Our, balanced)\033[0m\n");
    }

    free(A);
    free(csrA);
    free(B);
    free(C_gemmex);
    free(C_comparing);
    fprintf(stdout, "GemmEx : \033[36m[%lf] ms\033[0m\n", e_gemmex);
    fprintf(stdout, "cuSPARSE: \033[36m[%lf] ms\033[0m \n", e_cusparse);
    fprintf(stdout, "our: \033[36m[%lf] ms\033[0m \n", e_our);
}