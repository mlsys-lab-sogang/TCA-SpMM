#include "gemm_cublas.cuh"

void perform_GemmEx(half *A, half *B, float *C, int M, int N, int K, int n_iter, double *elapsed = NULL)
{
    warm_up_gpu_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();

    cublasHandle_t handle;

    float measured_time = 0.0;
    float alloc_time, h2d_time, d2h_time;
    alloc_time = h2d_time = d2h_time = 0.0;
    half *dA, *dB;
    float *dC;
    size_t total_mem = 0;

    float alpha = 1.0f;
    float beta = 0.0f;
    cudaDataType_t dtypeA, dtypeB, dtypeC;
    dtypeA = dtypeB = CUDA_R_16F;
    dtypeC = CUDA_R_32F;
    cublasComputeType_t comptype;
    comptype = CUBLAS_COMPUTE_32F;
    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));
    VERBOSE_PUTS("Preparing Resources...");
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMalloc((void **)&dA, sizeof(A[0]) * M * K));
    total_mem += sizeof(A[0]) * M * K;
    CHECK_CUDA(cudaMalloc((void **)&dB, sizeof(B[0]) * K * N));
    total_mem += sizeof(B[0]) * K * N;
    CHECK_CUDA(cudaMalloc((void **)&dC, sizeof(C[0]) * M * N));
    total_mem += sizeof(C[0]) * M * N;

    CHECK_CUBLAS(cublasCreate(&handle));
    // CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&alloc_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", alloc_time);

    VERBOSE_PUTS("H2D Memcpy...");
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUBLAS(cublasSetVector(M * K, sizeof(dA[0]), A, 1, dA, 1));
    CHECK_CUBLAS(cublasSetVector(K * N, sizeof(dB[0]), B, 1, dB, 1));
    CHECK_CUBLAS(cublasSetVector(M * N, sizeof(dC[0]), C, 1, dC, 1));

    // CHECK_CUDA(cudaMemcpy(dA, A, sizeof(A[0]) * M * K, cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(dC, B, sizeof(B[0]) * K * N, cudaMemcpyHostToDevice));
    // CHECK_CUDA(cudaMemcpy(dC, C, sizeof(C[0]) * M * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&h2d_time, start, end));

    VERBOSE_PRINT("Done %fms taken\n", h2d_time);

    for (int i = 0; i < 5; i++)
    {
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K,
                                  &alpha,
                                  dB, dtypeB, N,
                                  dA, dtypeA, K,
                                  &beta,
                                  dC, dtypeC, N,
                                  comptype, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    VERBOSE_PUTS("Performs GemmEx...");
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < n_iter; i++)
    {
        // obtain C^T, instead of C, to utilize col-major system.
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  N, M, K,
                                  &alpha,
                                  dB, dtypeB, N,
                                  dA, dtypeA, K,
                                  &beta,
                                  dC, dtypeC, N,
                                  comptype, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));

    CHECK_CUDA(cudaEventElapsedTime(&measured_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", measured_time);
    measured_time = measured_time / (float)n_iter;

    VERBOSE_PUTS("D2H  Memcpy...");
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUBLAS(cublasGetVector(M * N, sizeof(C[0]), dC, 1, C, 1));
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&d2h_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", d2h_time);

    if (elapsed != NULL)
    {
        *elapsed = (double)measured_time;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(end));

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
}