#include "our_spmm.cuh"
#include <cuda_pipeline.h>
template <typename T>
static __inline__ __device__ T warp_reduce_sum(T value)
{
    /* aggregate all value that each thread within a warp holding.*/
    T ret = value;

    for (int w = 1; w < warpSize; w = w << 1)
    {
        T tmp = __shfl_xor_sync(0xffffffff, ret, w);
        ret += tmp;
    }
    return ret;
}
static __device__ void flush_output(float *output, float *output_buffer, int output_buffer_size, int warpid, int laneid, int num_warps_per_row)
{
    for (int i = 4 * (warpid * warpSize + laneid); i < output_buffer_size; i += 4 * num_warps_per_row * warpSize)
    {
        reinterpret_cast<float4 *>(&output[i])[0] = reinterpret_cast<float4 *>(&output_buffer[i])[0];
        reinterpret_cast<float4 *>(&output_buffer[i])[0] = float4{0, 0, 0, 0};
    }
}

template <const int output_tile_width>
static __device__ void multirow_per_tb(half *spmat_val, const int *row_offset, int *col_idx, half *B, float *C,
                                       const int M, const int N, const int K, const float alpha, const float beta,
                                       int target_row, int warpid_current_row, int no_warps_within_row,
                                       half *A_val_scratch, half *B_scratch, float *output_buffer, int output_buffer_size)
{
    int laneid;
    asm("mov.s32 %0, %laneid;" : "=r"(laneid));
    float *output_row = &C[N * target_row];

    uint32_t reg_a[2]; // reg_a fetch B
    uint32_t reg_b[1]; // reg_b fetch A
    float reg_c[4];

    half2 *a = reinterpret_cast<half2 *>(reg_a);
    half2 *b = reinterpret_cast<half2 *>(reg_b);

    int row_start = row_offset[target_row];
    int sparse_row_nnz = row_offset[target_row + 1] - row_start;

    int2 fetching_idx;
    fetching_idx.x = warpid_current_row * 8 + 2 * (laneid / 8);
    fetching_idx.y = fetching_idx.x + 1;
    int2 fetching_col = {-1, -1};

    half2 Aval_tmp = {0, 0};
    int fetching_col_lane = 8 * (laneid % 8);
    if (fetching_idx.x < sparse_row_nnz)
    {
        fetching_col.x = col_idx[row_start + fetching_idx.x] * N + fetching_col_lane;
        Aval_tmp.x = spmat_val[row_start + fetching_idx.x];
    }
    if (fetching_idx.y < sparse_row_nnz)
    {
        fetching_col.y = col_idx[row_start + fetching_idx.y] * N + fetching_col_lane;
        Aval_tmp.y = spmat_val[row_start + fetching_idx.y];
    }
    reinterpret_cast<half2 *>(&A_val_scratch[fetching_idx.x])[0] = Aval_tmp;
    output_buffer_size = min(output_buffer_size, N);

    reg_a[0] = reg_a[1] = 0;
    __syncthreads();
    b[0] = reinterpret_cast<half2 *>(&A_val_scratch[2 * laneid])[0];

    int filled_buffer = 0;
    int register_choice = (laneid / 4) % 2;
    if (no_warps_within_row == 1)
        register_choice = 0;
    int B_scratch_leading_dim = 8 * no_warps_within_row;
    int invariant_offset = (16 / no_warps_within_row) * warpid_current_row + laneid / (4 * no_warps_within_row);
    int n_shuffle = 0;
    while ((no_warps_within_row >> n_shuffle) > 1)
        n_shuffle++;
    uint4 fetch_buffer1;
    uint4 fetch_buffer2;
    int first_transpose_idx = fetching_col_lane * (B_scratch_leading_dim) + (8 * warpid_current_row + 2 * (laneid / 8));

    int shm_row_step = 64 / B_scratch_leading_dim;
    bool writing_lane = (laneid % (4 * no_warps_within_row) == 0);
    for (int tile = 0; tile < N; tile += output_tile_width)
    {
        fetch_buffer1 = uint4{0, 0, 0, 0};
        fetch_buffer2 = uint4{0, 0, 0, 0};
        if (fetching_col.x != -1)
        {
            fetch_buffer1 = reinterpret_cast<uint4 *>(&B[fetching_col.x + tile])[0];
        }
        if (fetching_col.y != -1)
        {
            fetch_buffer2 = reinterpret_cast<uint4 *>(&B[fetching_col.y + tile])[0];
        }
        int transpose_idx = first_transpose_idx;

        for (int st = 0; st < 8; st++)
        {
            half2 tmp;
            tmp.x = reinterpret_cast<half *>(&fetch_buffer1)[st];
            tmp.y = reinterpret_cast<half *>(&fetch_buffer2)[st];
            reinterpret_cast<half2 *>(&B_scratch[transpose_idx])[0] = tmp;
            transpose_idx += B_scratch_leading_dim;
        }

        __syncthreads();
        float2 val;
        for (int idx = 0; idx < 4; idx++)
        {
            reinterpret_cast<float4 *>(&reg_c[0])[0] = float4{0, 0, 0, 0};
            int offset = B_scratch_leading_dim * (16 * idx + warpid_current_row * (16 / no_warps_within_row)) + 2 * laneid;
            a[0] = reinterpret_cast<half2 *>(&B_scratch[offset])[0];
            a[1] = reinterpret_cast<half2 *>(&B_scratch[offset + shm_row_step * B_scratch_leading_dim])[0];
            asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
                         "{ %0, %1, %2, %3},"
                         "{ %4, %5 },"
                         "{ %6 },"
                         "{ %0, %1, %2, %3};\n"
                         : "+f"(reg_c[0]), "+f"(reg_c[1]), "+f"(reg_c[2]), "+f"(reg_c[3])
                         : "r"(reg_a[0]), "r"(reg_a[1]),
                           "r"(reg_b[0]));
            val.x = reg_c[register_choice];
            val.y = reg_c[register_choice + 2];

            int shuffle_offset = 4;

            for (int n = 0; n < n_shuffle; n++)
            {
                val.x += __shfl_down_sync(0xffffffff, val.x, shuffle_offset);
                val.y += __shfl_down_sync(0xffffffff, val.y, shuffle_offset);
                shuffle_offset = 2 * shuffle_offset + ((n + 1) % 2);
            }

            if (no_warps_within_row == 8 && laneid == 0)
            {

                reinterpret_cast<float2 *>(&output_buffer[filled_buffer + 16 * idx + 2 * warpid_current_row])[0] = val;
            }
            else if (writing_lane)
            {
                output_buffer[filled_buffer + 16 * idx + invariant_offset] = val.x;
                output_buffer[filled_buffer + 16 * idx + invariant_offset + (8 / no_warps_within_row)] = val.y;
            }
        }
        filled_buffer += 64;
        __syncthreads();
        if (filled_buffer == output_buffer_size)
        {
            flush_output(output_row, output_buffer, output_buffer_size, warpid_current_row, laneid, no_warps_within_row);

            filled_buffer = 0;
            output_row = &output_row[output_buffer_size];
        }
    }
}
template <const int output_tile_width>
static __device__ void multitb_per_row(half *spmat_val, const int *row_offset, int *col_idx, half *B, float *C,
                                       const int M, const int N, const int K, const float alpha, const float beta,
                                       int target_row, int warpid_current_row, int no_warps_within_row, int col_start, int col_end,
                                       half *A_val_scratch, half *B_scratch, float *output_buffer, int output_buffer_size)
{
    int laneid;
    int warpid = threadIdx.x / warpSize;
    int num_warps = blockDim.x / warpSize;
    asm("mov.s32 %0, %laneid;" : "=r"(laneid));

    float *output_row = &C[N * target_row + col_start];

    uint32_t reg_a[2]; // reg_a fetch B
    uint32_t reg_b[1]; // reg_b fetch A
    float reg_c[16];

    half2 *a = reinterpret_cast<half2 *>(reg_a);
    half2 *b = reinterpret_cast<half2 *>(reg_b);

    int row_start = row_offset[target_row];
    int sparse_row_nnz = row_offset[target_row + 1] - row_start;

    int2 fetching_idx;
    int2 fetching_col;
    half2 Aval_tmp;

    int B_scratch_leading_dim = 64;

    uint4 fetch_buffer1;
    uint4 fetch_buffer2;

    int fetching_col_lane = 8 * (laneid % 8);
    int filled_buffer = 0;
    int register_choice = (laneid / 4) % 2;
    int first_transpose_idx = fetching_col_lane * (B_scratch_leading_dim) + (8 * warpid + 2 * (laneid / 8));

    output_buffer_size = min(output_buffer_size, col_end - col_start);

    // printf("X\n");
    for (int tile = col_start; tile < col_end; tile += output_tile_width)
    {
        for (int i = 0; i < 16; i += 4)
        {
            reinterpret_cast<float4 *>(&reg_c[i])[0] = float4{0, 0, 0, 0};
        }
        for (int nnz_count = 0; nnz_count < sparse_row_nnz; nnz_count += 64)
        {
            fetching_col = {-1, -1};

            fetching_idx.x = nnz_count + warpid * 8 + 2 * (laneid / 8);
            fetching_idx.y = fetching_idx.x + 1;
            Aval_tmp = half2{0, 0};
            fetch_buffer1 = uint4{0, 0, 0, 0};
            fetch_buffer2 = uint4{0, 0, 0, 0};

            if (fetching_idx.x < sparse_row_nnz)
            {
                fetching_col.x = col_idx[row_start + fetching_idx.x] * N + tile + fetching_col_lane;
                Aval_tmp.x = spmat_val[row_start + fetching_idx.x];

                if (fetching_col.x < K * N)
                    fetch_buffer1 = reinterpret_cast<uint4 *>(&B[fetching_col.x])[0];
            }
            if (fetching_idx.y < sparse_row_nnz)
            {
                fetching_col.y = col_idx[row_start + fetching_idx.y] * N + tile + fetching_col_lane;
                Aval_tmp.y = spmat_val[row_start + fetching_idx.y];
                if (fetching_col.y < K * N)
                    fetch_buffer2 = reinterpret_cast<uint4 *>(&B[fetching_col.y])[0];
            }
            int transpose_idx = first_transpose_idx;
            for (int st = 0; st < 8; st++)
            {
                half2 tmp;
                tmp.x = reinterpret_cast<half *>(&fetch_buffer1)[st];
                tmp.y = reinterpret_cast<half *>(&fetch_buffer2)[st];
                reinterpret_cast<half2 *>(&B_scratch[transpose_idx])[0] = tmp;

                transpose_idx += B_scratch_leading_dim;
            }

            reinterpret_cast<half2 *>(&A_val_scratch[fetching_idx.x - nnz_count])[0] = Aval_tmp;

            a[0] = a[1] = b[0] = half2{0, 0};
            __syncthreads();
            b[0] = reinterpret_cast<half2 *>(&A_val_scratch[2 * laneid])[0];
            for (int idx = 0; idx < 4; idx++)
            {
                int offset = B_scratch_leading_dim * (16 * idx + warpid * (16 / num_warps)) + 2 * laneid;
                a[0] = reinterpret_cast<half2 *>(&B_scratch[offset])[0];
                a[1] = reinterpret_cast<half2 *>(&B_scratch[offset + B_scratch_leading_dim])[0];
                asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
                             "{ %0, %1, %2, %3},"
                             "{ %4, %5 },"
                             "{ %6 },"
                             "{ %0, %1, %2, %3};\n"
                             : "+f"(reg_c[4 * idx]), "+f"(reg_c[4 * idx + 1]), "+f"(reg_c[4 * idx + 2]), "+f"(reg_c[4 * idx + 3])
                             : "r"(reg_a[0]), "r"(reg_a[1]),
                               "r"(reg_b[0]));
            }
        }
        float2 val;
        for (int idx = 0; idx < 4; idx++)
        {
            val.x = reg_c[4 * idx + register_choice];
            val.y = reg_c[4 * idx + register_choice + 2];

            val.x += __shfl_down_sync(0xffff0000, val.x, 18); // t0<-t18, t4 <-t22, t9<-t27, t13<-t31
            val.y += __shfl_down_sync(0xffff0000, val.y, 18); // t0<-t18, t4 <-t22, t9<-t27, t13<-t31
            val.x += __shfl_down_sync(0xff000000, val.x, 9);  // t0<-t9, t4 <-t13
            val.y += __shfl_down_sync(0xff000000, val.y, 9);  // t0<-t9, t4 <-t13
            val.x += __shfl_down_sync(0xf0000000, val.x, 4);  // t0<-t4
            val.y += __shfl_down_sync(0xf0000000, val.y, 4);  // t0<-t4

            if (laneid == 0)
            {
                reinterpret_cast<float2 *>(&output_buffer[filled_buffer + 16 * idx + 2 * warpid])[0] = val;
            }
            __syncthreads();
        }
        // __syncthreads();
        filled_buffer += output_tile_width;
        if (filled_buffer == output_buffer_size)
        {
            flush_output(output_row, output_buffer, output_buffer_size, warpid, laneid, no_warps_within_row);

            filled_buffer = 0;
            output_row = &output_row[output_buffer_size];
            output_buffer_size = min(output_buffer_size, col_end - tile - output_tile_width);
        }
    }
}
template <const int output_tile_width>
static __global__ void
    __launch_bounds__(256)
        kernel_our_spmm_balanced(half *spmat_val, const int *row_offset, int *col_idx, half *B, float *C,
                                 const int M, const int N, const int K, const float alpha, const float beta,
                                 int *num_warps_per_row, int *row_idx_for_each_warp, int *warpid_within_row)
{
    extern __shared__ uint32_t shmem[];

    int num_warps = blockDim.x / warpSize;
    int local_warpid = threadIdx.x / warpSize;
    int global_warpid = num_warps * blockIdx.x + local_warpid;

    int target_row = row_idx_for_each_warp[global_warpid];
    if (target_row == -1)
        return;
    int no_warps_within_row = num_warps_per_row[target_row];
    int warpid_current_row = warpid_within_row[global_warpid];
    int tb_idx_within_row = warpid_current_row / num_warps;
    int no_tb_within_row = no_warps_within_row / num_warps;
    int inter_tb_stride = 64 * (int)ceil((float)(N / no_tb_within_row) / 64.0f);
    int col_start = tb_idx_within_row * inter_tb_stride;
    int col_end = (tb_idx_within_row + 1) * inter_tb_stride;
    if (tb_idx_within_row == no_tb_within_row - 1)
        col_end = N;
    half *A_val_scratch = (half *)shmem;
    float *output_buffer = (float *)&A_val_scratch[num_warps * 8];
    half *B_scratch = (half *)&output_buffer[num_warps * 64];
    for (int i = 4 * threadIdx.x; i < 64 * num_warps; i += 4 * blockDim.x)
    {
        reinterpret_cast<uint4 *>(&output_buffer[i])[0] = uint4{0, 0, 0, 0};
    }
    for (int i = 8 * threadIdx.x; i < num_warps * 8; i += 8 * blockDim.x)
    {
        reinterpret_cast<uint4 *>(&A_val_scratch[i])[0] = uint4{0, 0, 0, 0};
    }
    for (int i = 8 * threadIdx.x; i < 64 * 8 * num_warps; i += 8 * blockDim.x)
    {
        reinterpret_cast<uint4 *>(&B_scratch[i])[0] = uint4{0, 0, 0, 0};
    }
    int first_warpid_in_this_row = local_warpid - warpid_current_row + num_warps * tb_idx_within_row;
    float *exclusive_output_buffer = &output_buffer[first_warpid_in_this_row * 64];
    half *exclusive_Aval_scratch = &A_val_scratch[first_warpid_in_this_row * 8];
    half *exclusive_B_scratch = &B_scratch[first_warpid_in_this_row * 8 * 64];
    int output_buffer_size = min(num_warps * 64, no_warps_within_row * 64);

    __syncthreads();

    if (no_warps_within_row <= num_warps)
    {
        multirow_per_tb<output_tile_width>(spmat_val, row_offset, col_idx, B, C, M, N, K, alpha, beta, target_row, warpid_current_row, no_warps_within_row, exclusive_Aval_scratch, exclusive_B_scratch, exclusive_output_buffer, output_buffer_size);
    }
    else
    {
        multitb_per_row<output_tile_width>(spmat_val, row_offset, col_idx, B, C, M, N, K, alpha, beta, target_row, warpid_current_row, no_warps_within_row, col_start, col_end, A_val_scratch, B_scratch, output_buffer, output_buffer_size);
    }
}
static void workload_manager(CSR *A_csr, int num_warps_per_tb, int N, int *blockdim, int *num_blocks, int output_tile_width, size_t *shm_size, int **d_num_warps_per_row, int **d_row_idx_for_each_warp, int **d_warpid_within_row)
{
    if (num_warps_per_tb < 8 || num_warps_per_tb > 32)
    {
        printf("Invalid number of warps per thread block (required to be 8<= number of warps <= 32)");
        (*blockdim) = 0;
        (*num_blocks) = 0;
        (*shm_size) = 0;

        return;
    }
    (*blockdim) = num_warps_per_tb * WARP_SIZE;
    int *num_warps_per_row = (int *)malloc(sizeof(int) * (A_csr->nRow));
    std::vector<int> vec;

    for (int i = 0; i < A_csr->nRow; i++)
    {
        int row_nnz = A_csr->row_offset[i + 1] - A_csr->row_offset[i];
        int num_demanding_warp;
        if (row_nnz == 0)
        {
            num_demanding_warp = 0;
        }
        else if (row_nnz <= 8)
        {
            num_demanding_warp = 1;
        }
        else if (row_nnz <= 16)
        {
            num_demanding_warp = 2;
        }
        else if (row_nnz <= 32)
        {
            num_demanding_warp = 4;
        }
        else if (row_nnz <= 64)
        {
            num_demanding_warp = 8;
        }
        else
        {
            num_demanding_warp = min((int)ceil((float)row_nnz / 64.0f), N / 64) * num_warps_per_tb;
        }

        num_warps_per_row[i] = num_demanding_warp;
    }

    std::vector<int> row_idx_for_each_warp;
    std::vector<int> warpid_within_row;
    int *tb_idx_each_row = (int *)malloc(sizeof(int) * A_csr->nRow);
    memset(tb_idx_each_row, -1, sizeof(int) * A_csr->nRow);
    int tb_idx = 0;
    for (int i = 0; i < A_csr->nRow; i++)
    {
        int num_warps = num_warps_per_row[i];
        if (num_warps == 0 || tb_idx_each_row[i] != -1)
            continue;
        int num_tb;
        if (num_warps <= num_warps_per_tb)
        {
            num_tb = 1;
            tb_idx_each_row[i] = tb_idx;

            std::vector<int> warpids(num_warps_per_tb, -1);
            std::vector<int> rowidx(num_warps_per_tb, -1);
            for (int idx = 0; idx < num_warps; idx++)
            {
                warpids[idx] = idx;
                rowidx[idx] = i;
            }
            for (int j = i + 1; j < A_csr->nRow; j++)
            {

                int new_num_warps = num_warps_per_row[j];
                if (new_num_warps == 0)
                    continue;
                if (num_warps + new_num_warps <= num_warps_per_tb)
                {
                    tb_idx_each_row[j] = tb_idx;
                    for (int idx = num_warps; idx < num_warps + new_num_warps; idx++)
                    {
                        warpids[idx] = idx - num_warps;
                        rowidx[idx] = j;
                    }
                    num_warps += new_num_warps;
                }

                if (num_warps == num_warps_per_tb)
                {
                    break;
                }
            }
            warpid_within_row.insert(warpid_within_row.end(), warpids.begin(), warpids.end());
            row_idx_for_each_warp.insert(row_idx_for_each_warp.end(), rowidx.begin(), rowidx.end());
        }
        else
        {
            tb_idx_each_row[i] = tb_idx;
            num_tb = num_warps / num_warps_per_tb;
            std::vector<int> warpids(num_warps);
            std::vector<int> rowidx(num_warps, i);
            std::iota(warpids.begin(), warpids.end(), 0);
            warpid_within_row.insert(warpid_within_row.end(), warpids.begin(), warpids.end());
            row_idx_for_each_warp.insert(row_idx_for_each_warp.end(), rowidx.begin(), rowidx.end());
        }
        tb_idx += num_tb;
    }
    size_t shm_per_warp = 0;
    shm_per_warp += 8 * sizeof(half);      // fetching A.value
    shm_per_warp += 64 * 8 * sizeof(half); // fetching B
    shm_per_warp += 64 * sizeof(float);

    (*blockdim) = WARP_SIZE * num_warps_per_tb;
    (*num_blocks) = tb_idx;
    (*shm_size) = shm_per_warp * num_warps_per_tb;
    CHECK_CUDA(cudaMalloc((void **)d_num_warps_per_row, sizeof(int) * A_csr->nRow));
    CHECK_CUDA(cudaMalloc((void **)d_row_idx_for_each_warp, sizeof(int) * num_warps_per_tb * tb_idx));
    CHECK_CUDA(cudaMalloc((void **)d_warpid_within_row, sizeof(int) * num_warps_per_tb * tb_idx));

    CHECK_CUDA(cudaMemcpy(*d_num_warps_per_row, num_warps_per_row, sizeof(int) * A_csr->nRow, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(*d_row_idx_for_each_warp, &row_idx_for_each_warp[0], sizeof(int) * num_warps_per_tb * tb_idx, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(*d_warpid_within_row, &warpid_within_row[0], sizeof(int) * num_warps_per_tb * tb_idx, cudaMemcpyHostToDevice));

    free(num_warps_per_row);
}
void our_spmm_balanced(CSR *A_csr, half *B, float *C, int M, int N, int K, int n_iter, double *elapsed = NULL)
{
    warm_up_gpu_kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    float measured_time = 0.0;
    float alloc_time, h2d_time, d2h_time, conversion_time;
    alloc_time = h2d_time = d2h_time = conversion_time = 0.0;

    int *dcol_idx, *drow_offset;
    half *dA_val;
    half *dB;
    float *dC;
    size_t total_mem = 0;

    cudaEvent_t start, end;
    VERBOSE_PUTS("Mtrix Conversion...");
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&conversion_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", conversion_time);

    VERBOSE_PUTS("Preparing Resources...");
    CHECK_CUDA(cudaEventRecord(start));
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

    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&h2d_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", h2d_time);

    VERBOSE_PUTS("Performs our SpMM...");

    const int tile_size = 64;

    int num_block;
    int blockdim;
    size_t shm_size;
    int num_warp_per_tb = 8;
    int *d_num_warps_per_row = NULL;
    int *d_row_idx_for_each_warp = NULL;
    int *d_warpid_within_row = NULL;

    auto preprocess_start = std::chrono::high_resolution_clock::now();
    workload_manager(A_csr, num_warp_per_tb, N, &blockdim, &num_block, tile_size, &shm_size, &d_num_warps_per_row, &d_row_idx_for_each_warp, &d_warpid_within_row);
    auto preprocess_end = std::chrono::high_resolution_clock::now();

    float elapsed_preprocess = std::chrono::duration<float, std::milli>(preprocess_end - preprocess_start).count();
    printf("preprocess %fms taken\n", elapsed_preprocess);
    cudaDeviceSynchronize();
    // num_block = M;
    printf("num_block: %d blockdim: %d shm: %fKB\n", num_block, blockdim, (float)shm_size / 1024.0f);
    for (int i = 0; i < 5; i++)
    {
        kernel_our_spmm_balanced<tile_size><<<num_block, blockdim, shm_size>>>(dA_val, drow_offset, dcol_idx, dB, dC, M, N, K, 1.0f, 0.0f, d_num_warps_per_row, d_row_idx_for_each_warp, d_warpid_within_row);
    }
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < n_iter; i++)
    {
        kernel_our_spmm_balanced<tile_size><<<num_block, blockdim, shm_size>>>(dA_val, drow_offset, dcol_idx, dB, dC, M, N, K, 1.0f, 0.0f, d_num_warps_per_row, d_row_idx_for_each_warp, d_warpid_within_row);
    }
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&measured_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", measured_time);

    VERBOSE_PUTS("D2H  Memcpy...");
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDA(cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventElapsedTime(&d2h_time, start, end));
    VERBOSE_PRINT("Done %fms taken\n", d2h_time);

    measured_time = measured_time / (double)n_iter;
    if (elapsed != NULL)
    {
        *elapsed = (double)measured_time;
    }
    CHECK_CUDA(cudaFree(dcol_idx));
    CHECK_CUDA(cudaFree(drow_offset));
    CHECK_CUDA(cudaFree(dA_val));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    if (d_num_warps_per_row)
    {
        CHECK_CUDA(cudaFree(d_num_warps_per_row));
        CHECK_CUDA(cudaFree(d_row_idx_for_each_warp));
        CHECK_CUDA(cudaFree(d_warpid_within_row));
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(end));
}