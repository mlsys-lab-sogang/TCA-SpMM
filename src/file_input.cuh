#ifndef _FILE_INPUT_CUH
#define _FILE_INPUT_CUH
#include "general.cuh"
#include "arg_parse.h"
using namespace std;
void mtx_input_to_Dense(ifstream &fin, half **mat, Arguments *arg);
void mtx_input_to_CSR(ifstream &fin, CSR **csr_mat, Arguments *arg);
void smtx_input_to_Dense(ifstream &fin, half **mat, char *filename, int *nRow, int *nCol, int *nnz);
void smtx_input_to_CSR(ifstream &fin, CSR **csr_mat, Arguments *arg);
#endif