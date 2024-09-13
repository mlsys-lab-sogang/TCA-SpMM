#ifndef _MAT_FORMAT_CUH
#define _MAT_FORMAT_CUH

#include "general.cuh"

const char *MATTYPE_STRING[] = {
    "DENSE",
    "CSR",
    "COO"};
typedef enum _mat_format
{
    dense,
    csr,
    coo
} mat_format;

void format_conversion(mat_format input_format, mat_format output_format);

void mtx_input_to_Dense(FILE *file, half *mat);
void mtx_input_to_CSR(FILE *file, half *value, int *row_offset, int *col_idx);
void smtx_input_to_Dense(FILE *file, half *mat);
void smtx_input_to_CSR(FILE *file, half *value, int *row_offset, int *col_idx);
#endif