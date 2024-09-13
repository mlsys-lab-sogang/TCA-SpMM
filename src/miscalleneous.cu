#include "general.cuh"

void print_mat(float *mat, int nRow, int nCol)
{
    int print_lim_row = PRINT_LIMIT < nRow ? PRINT_LIMIT : nRow;
    int print_lim_col = PRINT_LIMIT < nCol ? PRINT_LIMIT : nCol;
    for (int r = 0; r < print_lim_row; r++)
    {
        for (int c = 0; c < print_lim_col; c++)
        {
            printf("%6.2f", mat[r * nCol + c]);
        }
        puts("");
    }
}
void print_mat(half *mat, int nRow, int nCol)
{
    int print_lim_row = PRINT_LIMIT < nRow ? PRINT_LIMIT : nRow;
    int print_lim_col = PRINT_LIMIT < nCol ? PRINT_LIMIT : nCol;
    for (int r = 0; r < print_lim_row; r++)
    {
        for (int c = 0; c < print_lim_col; c++)
        {
            printf("%6.2f", __half2float(mat[r * nCol + c]));
        }
        puts("");
    }
}
void convert_dense_to_csr(half *mat, CSR **descr, int nRow, int nCol)
{
    *descr = (CSR *)malloc(sizeof(CSR));
    int nnz;
    (*descr)->nRow = nRow;
    (*descr)->nCol = nCol;

    nnz = 0;
    for (int i = 0; i < nRow * nCol; i++)
    {
        nnz += (__half2float(mat[i]) != 0.0f);
    }
    // printf("nnz : %d\n", nnz);
    (*descr)->nnz = nnz;
    (*descr)->row_offset = (int *)malloc(sizeof(int) * (nRow + 1));
    (*descr)->col_idx = (int *)malloc(sizeof(int) * nnz);
    (*descr)->value = (half *)malloc(sizeof(half) * nnz);

    (*descr)->row_offset[0] = 0;
    int idx = 0;
    for (int r = 0; r < nRow; r++)
    {
        (*descr)->row_offset[r + 1] = (*descr)->row_offset[r];
        int c = 0;
        for (; c < nCol; c++)
        {
            int i = r * nCol + c;
            if (__half2float(mat[i]) != 0.0f)
            {
                (*descr)->value[idx] = mat[i];
                (*descr)->col_idx[idx] = c;
                (*descr)->row_offset[r + 1] += 1;
                idx++;
            }
        }
    }
}
void convert_csr_to_dense(CSR *descr, half **mat)
{
    *mat = (half *)malloc(sizeof(half) * (descr->nRow) * (descr->nCol));
    memset(*mat, __float2half(0), sizeof(half) * (descr->nRow) * (descr->nCol));
    int nRow = descr->nRow;
    int nCol = descr->nCol;
    for (int r = 0; r < nRow; r++)
    {
        for (int idx = descr->row_offset[r]; idx < descr->row_offset[r + 1]; idx++)
        {
            int c = descr->col_idx[idx];
            half v = descr->value[idx];
            (*mat)[r * nCol + c] = v;
        }
    }
}

void destroy_csr(CSR *descr)
{
    free(descr->col_idx);
    free(descr->row_offset);
    free(descr->value);
    free(descr);
}
void print_csr(CSR *descr)
{
    puts("row_index");
    for (int i = 0; i < descr->nRow; i++)
    {
        printf("%d ", (descr->row_offset)[i]);
    }
    puts("");

    puts("col_index");
    for (int i = 0; i < descr->nnz; i++)
    {
        printf("%d ", (descr->col_idx)[i]);
    }
    puts("");

    puts("value");
    for (int i = 0; i < descr->nnz; i++)
    {
        printf("%f ", __half2float((descr->value)[i]));
    }
    puts("");
}

void naive_gemm(half *A, half *B, float *C, int M, int N, int K)
{
    int m, n, k;
#pragma omp parallel for private(m, n, k) shared(A, B, C)
    for (m = 0; m < M; m++)
    {
        for (n = 0; n < N; n++)
        {
            float result = 0.0f;
            for (k = 0; k < K; k++)
            {
#if COUNT_OP == true
#pragma omp atomic
                num_MAC += 1;
#endif
                result += __half2float(A[m * K + k]) * __half2float(B[k * N + n]);
            }
            C[m * N + n] = result;
        }
    }
}
void print_succeed(bool succeed)
{
    if (succeed)
    {
        puts("\033[36mSucceed\033[0m");
    }
    else
    {
        puts("\033[31mFail\033[0m");
    }
}
__global__ void simple_transpose(half *Mat, half *Mat_t, int nRow, int nCol)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int working_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < nRow * nCol; i += working_threads)
    {
        half value = Mat[i];
        int source_row = i / nCol;
        int source_col = i % nCol;
        Mat_t[source_col * nRow + source_row] = value;
    }
}
__global__ void warm_up_gpu_kernel()
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}