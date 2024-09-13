#include "spmm_cusparse.cuh"
void perform_cuSPARSELT_24sparsity(half *A, half *B, float *C, int M, int N, int K, int num_iter, double *elapsed = NULL)
{
    cusparseLtHandle_t handle;
    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulDescriptor_t matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t plan;
    cudaStream_t stream = nullptr;

    cusparseComputeType compute_type = CUSPARSE_COMPUTE_32F;
    half *dA, *dB, *dA_compressed;

    half *dC;
    half *Ctmp;
    int *d_valid;
    size_t total_mem = 0;

    float measured_time = 0.0;
    float alloc_time, h2d_time, init_time, d2h_time;

    float alpha = 1.0f;
    float beta = 0.0f;
    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    VERBOSE_PUTS("Preparing Resources...");
    Ctmp = (half *)malloc(sizeof(half) * M * N);
    memset(Ctmp, __float2half(0.0f), sizeof(half) * M * N);
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMalloc((void **)&dA, M * K * sizeof(A[0])));
    CHECK_CUDA(cudaMalloc((void **)&dB, K * N * sizeof(B[0])));
    CHECK_CUDA(cudaMalloc((void **)&dC, M * N * sizeof(Ctmp[0])));
    CHECK_CUDA(cudaMalloc((void **)&d_valid, sizeof(int)));
    total_mem += sizeof(A[0]) * M * K;
    total_mem += sizeof(B[0]) * K * N;
    total_mem += sizeof(C[0]) * M * N;
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&alloc_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", alloc_time);

    VERBOSE_PUTS("H2D Memcpy...");
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(dA, A, M * K * sizeof(A[0]), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, K * N * sizeof(B[0]), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, Ctmp, M * N * sizeof(Ctmp[0]), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&h2d_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", h2d_time);

    VERBOSE_PUTS("Intializing cusparseLt...");
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUSPARSE(cusparseLtInit(&handle));
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&handle,
                                                      &matA, M, K, K, 16,
                                                      CUDA_R_16F,
                                                      CUSPARSE_ORDER_ROW,
                                                      CUSPARSELT_SPARSITY_50_PERCENT));
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle,
                                                 &matB, K, N, N, 16,
                                                 CUDA_R_16F,
                                                 CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&handle,
                                                 &matC, M, N, N, 16,
                                                 CUDA_R_16F,
                                                 CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&handle,
                                                  &matmul,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &matA, &matB, &matC, &matC,
                                                  compute_type));
    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&handle,
                                                    &alg_sel, &matmul,
                                                    CUSPARSELT_MATMUL_ALG_DEFAULT));

    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));

    CHECK_CUSPARSE(cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
                                        CUSPARSELT_PRUNE_SPMMA_STRIP, stream));
    CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,
                                             d_valid, stream));
    int is_valid;
    CHECK_CUDA(cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),
                               cudaMemcpyDeviceToHost, stream))
    CHECK_CUDA(cudaStreamSynchronize(stream))
    if (is_valid != 0)
    {
        VERBOSE_PRINT("!!!! The matrix has been pruned in a wrong way. "
                      "cusparseLtMatmul will not provide correct results\n");
        return;
    }
    size_t compressed_size, compressed_buffer_size;
    half *dA_compressedBuffer;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize(&handle, &plan,
                                                 &compressed_size,
                                                 &compressed_buffer_size));
    CHECK_CUDA(cudaMalloc((void **)&dA_compressed, compressed_size));
    CHECK_CUDA(cudaMalloc((void **)&dA_compressedBuffer, compressed_buffer_size));
    CHECK_CUSPARSE(cusparseLtSpMMACompress(&handle, &plan, dA, dA_compressed,
                                           dA_compressedBuffer, stream));
    int num_streams = 0;
    cudaStream_t *streams = nullptr;
    // CHECK_CUSPARSE(cusparseLtMatmulSearch(&handle, &plan, &alpha,
    //                                       dA_compressed, dB, &beta,
    //                                       dC, dC, nullptr,
    //                                       streams, num_streams));
    // // dC accumulates so reset dC for correctness check
    // CHECK_CUDA(cudaMemcpy(dC, Ctmp, M * N * sizeof(half), cudaMemcpyHostToDevice));
    int alg = 0;
    CHECK_CUSPARSE(cusparseLtMatmulAlgSetAttribute(
        &handle, &alg_sel,
        CUSPARSELT_MATMUL_ALG_CONFIG_ID,
        &alg, sizeof(alg)))

    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&init_time, start, end));

    VERBOSE_PRINT("Done %fms taken\n", init_time);

    VERBOSE_PUTS("Performs cuSPARSELT_24sparsity...");
    size_t workspace_size;
    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel));

    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&handle, &plan,
                                                &workspace_size));
    void *d_workspace;
    CHECK_CUDA(cudaMalloc((void **)&d_workspace, workspace_size));
    // Perform the matrix multiplication
    CHECK_CUDA(cudaEventRecord(start));
    for (int iter = 0; iter < num_iter; iter++)
    {
        CHECK_CUSPARSE(cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                                        &beta, dC, dC, d_workspace, streams,
                                        num_streams));
    }
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&measured_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", measured_time);
    measured_time = measured_time / (float)num_iter;

    VERBOSE_PUTS("D2H  Memcpy...");
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(Ctmp, dC, sizeof(half) * M * N, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&d2h_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", d2h_time);

    int i;
    // #pragma omp parallel for private(i) shared(C, Ctmp)
    for (i = 0; i < M * N; i++)
    {
        C[i] = __half2float(Ctmp[i]);
    }

    if (elapsed != NULL)
    {
        *elapsed = (double)measured_time;
    }

    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matA));
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matB));
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&matC));
    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&plan));
    CHECK_CUSPARSE(cusparseLtDestroy(&handle));
    free(Ctmp);
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dA_compressed));
    CHECK_CUDA(cudaFree(dA_compressedBuffer));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
}

void perform_cuSPARSE_BlockSpMM(half *A, half *B, float *C, int M, int N, int K, int num_iter, int ell_block_size, double *elapsed = NULL)
{
    warm_up_gpu_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    cusparseHandle_t handle = nullptr;
    float measured_time = 0.0;
    float alloc_time, h2d_time, init_time, d2h_time, conversion_time;
    cusparseSpMatDescr_t matA_BlockedEll;
    cusparseDnMatDescr_t matA, matB, matC;
    int ell_blocksize = ell_block_size;

    half *dA_columns;
    half *dA_values;

    half *dA, *dB;
    float *dC;

    float alpha = 1.0f;
    float beta = 0.0f;

    int ell_rows = ceil((float)M / (float)ell_blocksize);
    int origin_M = M;
    int origin_K = K;
    if (ell_rows * ell_block_size != M)
    {
        M = ell_rows * ell_block_size;
    }
    if (K % ell_block_size != 0)
    {
        K = ell_block_size * ceil((float)K / (float)ell_block_size);
    }

    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    VERBOSE_PUTS("Preparing Resources...");
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMalloc((void **)&dA, M * K * sizeof(A[0])));
    CHECK_CUDA(cudaMalloc((void **)&dB, K * N * sizeof(B[0])));
    CHECK_CUDA(cudaMalloc((void **)&dC, M * N * sizeof(C[0])));

    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&alloc_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", alloc_time);
    CHECK_CUDA(cudaMemcpy(dA, A, sizeof(half) * M * origin_K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, sizeof(half) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, C, sizeof(float) * M * N, cudaMemcpyHostToDevice));

    VERBOSE_PUTS("H2D Memcpy...");
    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&h2d_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", h2d_time);

    VERBOSE_PUTS("Intializing cusparse...");
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUSPARSE(cusparseCreate(&handle));

    CHECK_CUSPARSE(cusparseCreateDnMat(&matA, M, K, K,
                                       dA, CUDA_R_16F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, K, N, N,
                                       dB, CUDA_R_16F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, M, N, N,
                                       dC, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&init_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", init_time);

    VERBOSE_PUTS("matrix conversion...");
    CHECK_CUDA(cudaEventRecord(start));

    int num_ell_col_blocks = 0;
    std::vector<std::vector<int>> nnz(ell_rows, std::vector<int>(ceil((float)K / (float)ell_blocksize), 0));

#pragma omp parallel for num_threads(12)
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < origin_K; j++)
        {
            int is_nz = (int)(__half2float(A[i * K + j]) != 0.0f);
            nnz[i / ell_blocksize][j / ell_blocksize] += is_nz;
        }
    }
    // for (int i = 0; i < M / ell_blocksize; i++)
    // {
    //     //VERBOSE_PRINT("%d th row: %d ~ %d \n", i, start_pos, end_pos);
    //     for (int j = 0; j < K / ell_blocksize; j++)
    //     {
    //        VERBOSE_PRINT("%d ", nnz[i][j]);
    //     }
    //     VERBOSE_PUTS("");
    // }
    num_ell_col_blocks = 0;
    std::vector<std::vector<int>> dense_tiles(ell_rows);

    for (int i = 0; i < M / ell_blocksize; i++)
    {
        int cols_tmp = 0;
        for (int j = 0; j < K / ell_blocksize; j++)
        {
            cols_tmp += 0 < nnz[i][j];
        }
        num_ell_col_blocks = max(num_ell_col_blocks, cols_tmp);
    }
    int *h_ellcolidx = (int *)malloc(sizeof(int) * ell_rows * num_ell_col_blocks);
    // #pragma omp parallel for num_threads(12)
    for (int i = 0; i < M / ell_blocksize; i++)
    {
        int ell_col_idx = 0;
        for (int j = 0; j < K / ell_blocksize; j++)
        {
            int is_dense = 0 < nnz[i][j];
            if (is_dense)
            {
                h_ellcolidx[i * num_ell_col_blocks + ell_col_idx] = j;
                ell_col_idx++;
            }
        }
        for (; ell_col_idx < num_ell_col_blocks; ell_col_idx++)
        {
            h_ellcolidx[i * num_ell_col_blocks + ell_col_idx] = -1;
        }
    }

    int ell_cols = num_ell_col_blocks * ell_blocksize;
    half *A_val = (half *)malloc(sizeof(half) * ell_cols * M);
    memset(A_val, __float2half(0.0f), sizeof(half) * ell_cols * M);
    CHECK_CUDA(cudaMalloc((void **)&dA_columns, sizeof(int) * ell_rows * num_ell_col_blocks));
    CHECK_CUDA(cudaMalloc((void **)&dA_values, ell_cols * M * sizeof(half)));
    for (int i = 0; i < ell_cols * M; i++)
    {
        int brow = (i / ell_cols) / ell_blocksize; //(i / ell_blocksize) / num_ell_col_blocks;
        int bcol = (i % ell_cols) / ell_blocksize; //(i / ell_blocksize) % num_ell_col_blocks;
        // VERBOSE_PRINT("i:%d block:%d %d \n", i, brow, bcol);

        if (h_ellcolidx[brow * num_ell_col_blocks + bcol] == -1)
        {
            A_val[i] = __float2half(0.0f);
            continue;
        }
        int row_within_block = ((i / ell_cols) % ell_blocksize);
        int col_within_block = ((i % ell_cols) % ell_blocksize);
        int block_start = brow * K * ell_blocksize + h_ellcolidx[brow * num_ell_col_blocks + bcol] * ell_blocksize;
        // VERBOSE_PRINT("i:%d block:%d %d in block:%d %d ::%d\n", i, brow, bcol, row_within_block, col_within_block, block_start);
        A_val[i] = A[block_start + row_within_block * K + col_within_block];
    }

    CHECK_CUDA(cudaMemcpy(dA_columns, h_ellcolidx, sizeof(int) * ell_rows * num_ell_col_blocks, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, A_val, sizeof(half) * ell_cols * M, cudaMemcpyHostToDevice));

    CHECK_CUSPARSE(cusparseCreateBlockedEll(&matA_BlockedEll, M, K,
                                            ell_blocksize, ell_cols,
                                            dA_columns, dA_values,
                                            CUSPARSE_INDEX_32I,
                                            CUSPARSE_INDEX_BASE_ZERO,
                                            CUDA_R_16F));
    // CHECK_CUSPARSE(cusparseDenseToSparse_bufferSize(handle,
    //                                                 matA, matA_BlockedEll,
    //                                                 CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
    //                                                 &bufferSize_d2s));
    // CHECK_CUDA(cudaMalloc(&dBuffer_d2s, bufferSize_d2s));
    // CHECK_CUSPARSE(cusparseDenseToSparse_analysis(handle, matA, matA_BlockedEll,
    //                                               CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
    //                                               dBuffer_d2s));
    // CHECK_CUSPARSE(cusparseDenseToSparse_convert(handle, matA, matA_BlockedEll,
    //                                              CUSPARSE_DENSETOSPARSE_ALG_DEFAULT,
    //                                              dBuffer_d2s));

    // CHECK_CUDA(cudaFree(dBuffer_d2s));
    // half *A_val = (half *)malloc(sizeof(half) * ell_cols * M);
    // CHECK_CUDA(cudaMemcpy(A_val, dA_values, sizeof(half) * ell_cols * M, cudaMemcpyDeviceToHost));
    // VERBOSE_PRINT("%d\n\n\n", num_ell_col_blocks);
    // print_mat(A, M, K);
    // VERBOSE_PUTS("\n\n");

    // for (int i = 0; i < ell_rows * num_ell_col_blocks; i++)
    // {
    //    VERBOSE_PRINT("%d ", h_ellcolidx[i]);
    //     if (i % num_ell_col_blocks == num_ell_col_blocks - 1)
    //         VERBOSE_PUTS("");
    // }
    // VERBOSE_PUTS("\n\n");
    // for (int i = 0; i < ell_cols * M; i++)
    // {
    //     // if (__half2float(A_val[i]) != 0.0f)
    //    VERBOSE_PRINT("%.2f ", __half2float(A_val[i]));
    //     if (i % ell_cols == ell_cols - 1)
    //         VERBOSE_PUTS("");
    // }
    void *dBuffer;
    size_t bufferSize;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA_BlockedEll, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));

    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&conversion_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", conversion_time);

    VERBOSE_PUTS("Warmup Block SpMM...");
    for (int i = 0; i < 5; i++)
    {
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA_BlockedEll, matB, &beta, matC, CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    VERBOSE_PUTS("Compute by cuSPARSE Block SpMM...");
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_iter; i++)
    {
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA_BlockedEll, matB, &beta, matC, CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
    }
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&measured_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", measured_time);
    measured_time = measured_time / (float)num_iter;

    VERBOSE_PUTS("D2H  Memcpy...");
    CHECK_CUDA(cudaEventRecord(start));
    if (M == origin_M)
        (cudaMemcpy(C, dC, M * N * sizeof(float),
                    cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&d2h_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", d2h_time);

    if (elapsed != NULL)
    {
        *elapsed = (double)measured_time;
    }
    cusparseDestroySpMat(matA_BlockedEll);
    cusparseDestroyDnMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);
    cudaFree(dBuffer);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dA_columns);
    cudaFree(dA_values);
    free(h_ellcolidx);
    free(A_val);
}
void perform_cuSPARSE_SpMM(half *A, half *B, float *C, int M, int N, int K, int num_iter, double *elapsed)
{
    warm_up_gpu_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    float measured_time = 0.0;
    float alloc_time, h2d_time, d2h_time, conversion_time;
    alloc_time = h2d_time = d2h_time = conversion_time = 0.0;
    int *dcol_idx, *drow_offset;
    half *dA_val;
    half *dB;
    float *dC;
    size_t total_mem = 0;
    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUSPARSE(cusparseCreate(&handle))
    CSR *A_csr;
    convert_dense_to_csr(A, &A_csr, M, K);
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&conversion_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", conversion_time);

    VERBOSE_PUTS("Preparing Resources...");
    CHECK_CUDA(cudaEventRecord(start));
    // CHECK_CUDA(cudaMalloc((void **)&dA, sizeof(A[0]) * M * K));
    // total_mem += sizeof(A[0]) * M * K;
    CHECK_CUDA(cudaMalloc((void **)&dA_val, sizeof(A_csr->value[0]) * A_csr->nnz));
    CHECK_CUDA(cudaMalloc((void **)&dcol_idx, sizeof(A_csr->col_idx[0]) * A_csr->nnz));
    CHECK_CUDA(cudaMalloc((void **)&drow_offset, sizeof(A_csr->row_offset[0]) * (A_csr->nRow + 1)));

    CHECK_CUDA(cudaMalloc((void **)&dB, sizeof(B[0]) * K * N));
    total_mem += sizeof(B[0]) * K * N;
    CHECK_CUDA(cudaMalloc((void **)&dC, sizeof(C[0]) * M * N));
    total_mem += sizeof(C[0]) * M * N;
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&alloc_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", alloc_time);

    VERBOSE_PUTS("H2D Memcpy...");
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(dA_val, A_csr->value, sizeof(A_csr->value[0]) * A_csr->nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dcol_idx, A_csr->col_idx, sizeof(A_csr->col_idx[0]) * A_csr->nnz, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(drow_offset, A_csr->row_offset, sizeof(A_csr->row_offset[0]) * (A_csr->nRow + 1), cudaMemcpyHostToDevice));

    // CHECK_CUDA(cudaMemcpy(dA, A, sizeof(A[0]) * M * K, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, B, sizeof(B[0]) * K * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, C, sizeof(C[0]) * M * N, cudaMemcpyHostToDevice));

    CHECK_CUSPARSE(cusparseCreateCsr(&matA, M, K, A_csr->nnz,
                                     drow_offset, dcol_idx, dA_val,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F));
    // Create dense matrix B
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, K, N, N, dB,
                                       CUDA_R_16F, CUSPARSE_ORDER_ROW));
    // Create dense matrix C
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, M, N, N, dC,
                                       CUDA_R_32F, CUSPARSE_ORDER_ROW));
    float alpha = 1.0f;
    float beta = 0.0f;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&h2d_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", h2d_time);

    VERBOSE_PUTS("Warm up cuSPARSE...");
    for (int i = 0; i < num_iter; i++)
    {
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    VERBOSE_PUTS("Compute by cuSPARSE...");
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < num_iter; i++)
    {
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
    }
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&measured_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", measured_time);
    measured_time = measured_time / (float)num_iter;

    VERBOSE_PUTS("D2H  Memcpy...");
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(C, dC, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&d2h_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", d2h_time);

    if (elapsed != NULL)
    {
        *elapsed = (double)measured_time;
    }
    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);
    cudaFree(dBuffer);
    cudaFree(dA_val);
    cudaFree(drow_offset);
    cudaFree(dcol_idx);
    cudaFree(dB);
    cudaFree(dC);
}