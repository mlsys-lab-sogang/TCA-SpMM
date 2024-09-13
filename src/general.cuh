#ifndef _GENERAL_CUH
#define _GENERAL_CUH
#include <stdio.h>
#include <cuda_fp16.h>
#include <random>
#include <mkl/mkl.h>
#include <chrono>
#define CHECK_CUDA(f)                                                                                             \
    {                                                                                                             \
        cudaError_t err = (f);                                                                                    \
        if (err != cudaSuccess)                                                                                   \
        {                                                                                                         \
            fprintf(stderr, "CUDA error at [%s : %d] %d %s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                         \
    }

// #define UNIT_MMA_M 16
// #define UNIT_MMA_N 8
// #define UNIT_MMA_K 8
#define DEFAULT_M 1024
#define DEFAULT_N 1024
#define DEFAULT_K 1024
#define DEFAULT_TOL 1e-3
#define PRINT_LIMIT 32
#define VERBOSE false
#define TARGET_SPARSITY 0.8f
#define N_ITER 10
#define WARP_SIZE 32

#define VERBOSE_PRINT(...)   \
    if (VERBOSE == true)     \
    {                        \
        printf(__VA_ARGS__); \
    }

#define VERBOSE_PUTS(...)  \
    if (VERBOSE == true)   \
    {                      \
        puts(__VA_ARGS__); \
    }

#define GPU_DEBUG(...)       \
    if (threadIdx.x == 0)    \
    {                        \
        printf(__VA_ARGS__); \
    }

#define COUNT_OP false
#if COUNT_OP == true
static int num_MAC = 0;
static int num_MMA = 0;
#endif
typedef enum entry_dtype
{
    fp16,
    fp32,
    fp64
} entry_dtype;

typedef enum unit_mma_type
{
    half_to_float_m16n8k8,
    half_to_float_m16n8k16,
    double_to_double_m8n8k4
} unit_mma_type;

void print_mat(float *mat, int nRow, int nCol);
void print_mat(half *mat, int nRow, int nCol);

template <typename T>
bool compare_mat(T *gold, T *comparing, int num_elem, float tolerance)
{
    bool is_half_entry = false;
    if (std::is_same<T, half>::value)
    {
        is_half_entry = true;
    }
    bool success = true;
    int count = 0;

    float max_disc = 0.0f;
    bool print = true;
    for (int i = 0; i < num_elem; i++)
    {
        float v1, v2;
        if (is_half_entry)
        {
            v1 = __half2float(gold[i]);
            v2 = __half2float(comparing[i]);
        }
        else
        {
            v1 = (float)gold[i];
            v2 = (float)comparing[i];
        }

        float disc = fabs(v1 - v2);

        float rel_disc = fabs(disc / v1);
        max_disc = max(rel_disc, max_disc);
        if (rel_disc >= tolerance && disc >= tolerance)
        {

            success = false;
            printf("Discrepancy occured in %dth element :: %f vs %f ...abs_disc : %f, rel_disc : %f %%\n", i, v1, v2, disc, rel_disc * 100.0);
            count++;
            if (print && count >= 10)
            {
                VERBOSE_PUTS("stop print");
                VERBOSE_PRINT("Maximum discrepancy = %f %%, tol = %f %%\n", max_disc * 100.0, tolerance * 100.0);
                print = false;
                return success;
            }
        }
        // else
        // {
        //     printf("%lf %lf %f %f %f\n", v1, v2, rel_disc, disc, tolerance);
        // }
    }
    VERBOSE_PRINT("Maximum discrepancy = %f %%, tol = %f %%\n", max_disc * 100.0, tolerance * 100.0);
    return success;
}

typedef struct CSR
{
    int nRow;
    int nCol;
    int nnz;
    int *row_offset;
    int *col_idx;
    half *value;

} CSR;
void convert_dense_to_csr(half *mat, CSR **descr, int nRow, int nCol);
void convert_csr_to_dense(CSR *descr, half **mat);
void destroy_csr(CSR *descr);
void print_csr(CSR *descr);

template <typename T>
void prepare_cpu_matrix(T **mat, size_t nRow, size_t nCol, float target_sparsity, bool leave_empty = false)
{
    *mat = (T *)malloc(sizeof(T) * nRow * nCol);
    if (leave_empty)
    {
        memset(*mat, 0, nRow * nCol * sizeof(T));
        return;
    }

#pragma omp parallel num_threads(20)
    {
        std::random_device rd;
        std::mt19937 e2(rd());
        std::uniform_real_distribution<> dist(0, 1);

        for (int i = 0; i < nRow * nCol; i++)
        {
            double elem;
            if (dist(e2) > target_sparsity)
            {
                elem = (double)dist(e2);
            }
            else
            {
                elem = 0.0;
            }
            if (std::is_same<T, half>::value)
            {
                // elem = elem * 4.0 - 2.0;
                (*mat)[i] = __double2half(elem);
            }
        }
    }
}
template <typename T>
void prepare_gpu_matrix(T *cpu_mat, T **gpu_mat, size_t n_elem, bool leave_empty = false)
{
    CHECK_CUDA(cudaMalloc(gpu_mat, sizeof(T) * n_elem));
    if (leave_empty)
    {
        CHECK_CUDA(cudaMemset(*gpu_mat, 0, sizeof(T) * n_elem));
        return;
    }
    CHECK_CUDA(cudaMemcpy(*gpu_mat, cpu_mat, sizeof(T) * n_elem, cudaMemcpyHostToDevice));
}

typedef struct _Blocked_Ell
{
    half *ellValue;
    int *ellColInd;
    int blocksize;
    int blocked_ell_columns;
    int blocked_ell_rows;

} Blocked_Ell;

void naive_gemm(half *A, half *B, float *C, int M, int N, int K);
void print_succeed(bool succeed);

__global__ void simple_transpose(half *Mat, half *Mat_t, int nRow, int nCol);
__global__ void warm_up_gpu_kernel();

#endif