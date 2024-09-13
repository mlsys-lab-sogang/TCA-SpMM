#include "file_input.cuh"
#include <string.h>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include <assert.h>
void mtx_input_to_Dense(ifstream &fin, half **mat, Arguments *arg)
{
    string line;
    int rows, cols, meta_nnz;
    while (getline(fin, line))
    {
        if (line[0] == '%')
            continue;
        else
        {
            stringstream meta_input(line);
            meta_input >> rows >> cols >> meta_nnz;
            break;
        }
    }
    map<int, vector<int>> dict;
    map<int, vector<half>> vals;

    int nzcount = 0;
    while (getline(fin, line))
    {
        int r, c;
        float v = 1.0f;
        stringstream input_string(line);
        input_string >> r >> c >> v;
        if (!(arg->input_is_zero_based))
        {
            r--;
            c--;
        }
        if (dict.find(r) == dict.end())
        {
            dict[r] = vector<int>();
            vals[r] = vector<half>();
        }

        dict[r].push_back(c);
        vals[r].push_back(__float2half(v));
        nzcount++;
    }

    assert(nzcount == meta_nnz);
    *mat = (half *)malloc(sizeof(half) * rows * cols);
    memset(*mat, __float2half(0.0f), rows * cols * sizeof(half));
    for (int i = 0; i < rows; i++)
    {
        auto row_pos = dict[i];
        for (int j = 0; j < row_pos.size(); j++)
        {
            int c = row_pos[j];
            half v = vals[i].at(j);

            (*mat)[i * cols + c] = v;
        }
    }
}
void mtx_input_to_CSR(ifstream &fin, CSR **csr_mat, Arguments *arg)
{
    string line;
    int rows, cols, meta_nnz;
    while (getline(fin, line))
    {
        if (line[0] == '%')
            continue;
        else
        {
            stringstream meta_input(line);
            meta_input >> rows >> cols >> meta_nnz;
            break;
        }
    }
    map<int, vector<int>> dict;
    map<int, vector<half>> vals;

    int nzcount = 0;
    while (getline(fin, line))
    {
        int r, c;
        float v;
        stringstream input_string(line);
        if (arg->pattern_only)
        {
            v = 1.0f;
            input_string >> r >> c;
        }
        else
        {
            v = 0.0f;
            input_string >> r >> c >> v;
        }
        if (!(arg->input_is_zero_based))
        {
            r--;
            c--;
        }
        if (dict.find(r) == dict.end())
        {
            dict[r] = vector<int>();
            vals[r] = vector<half>();
        }

        dict[r].push_back(c);
        vals[r].push_back(__float2half(v));
        nzcount++;
    }

    assert(nzcount == meta_nnz);
    *csr_mat = (CSR *)malloc(sizeof(CSR));
    (*csr_mat)->nRow = rows;
    (*csr_mat)->nCol = cols;
    (*csr_mat)->nnz = meta_nnz;
    (*csr_mat)->row_offset = (int *)malloc(sizeof(int) * (rows + 1));
    (*csr_mat)->col_idx = (int *)malloc(sizeof(int) * meta_nnz);
    (*csr_mat)->value = (half *)malloc(sizeof(half) * meta_nnz);

    (*csr_mat)->row_offset[0] = 0;

    int offset = 0;
    for (int i = 0; i < rows; i++)
    {
        auto row_pos = dict[i];
        (*csr_mat)->row_offset[i + 1] = (*csr_mat)->row_offset[i] + row_pos.size();
        std::copy(dict[i].begin(), dict[i].end(), (*csr_mat)->col_idx + offset);
        std::copy(vals[i].begin(), vals[i].end(), (*csr_mat)->value + offset);
        offset += row_pos.size();
    }
}
void smtx_input_to_Dense(ifstream &fin, half **mat, char *filename, int *nRow, int *nCol, int *nnz)
{
    string line;
    string buffer;
    map<int, vector<int>> dict;
    // header
    getline(fin, line);
    stringstream sin_meta(line);
    getline(sin_meta, buffer, ',');
    *nRow = stoi(buffer);
    getline(sin_meta, buffer, ',');
    *nCol = stoi(buffer);
    getline(sin_meta, buffer, ',');
    *nnz = stoi(buffer);

    int *rowptr = new int[(*nRow) + 1];
    int *colidx = new int[(*nnz)];
    half *values = new half[(*nnz)];
    std::vector<int> vec_colidx((*nnz));

    getline(fin, line);
    stringstream sin_row(line);
    int offset = 0, idx = 0;
    while (getline(sin_row, buffer, ' '))
    {
        offset = stoi(buffer);
        rowptr[idx++] = offset;
        // vec_rowptr[idx++] = offset;
    }
    int c;
    idx = 0;
    offset = 0;

    (*mat) = (half *)malloc(sizeof(half) * (*nRow) * (*nCol));
    memset((*mat), 0, sizeof(half) * (*nRow) * (*nCol));

    while (getline(fin, line))
    {
        stringstream sin_col(line);
        while (getline(sin_col, buffer, ' '))
        {
            c = stoi(buffer);

            vec_colidx[offset++] = c;
        }
    }
    std::copy(vec_colidx.begin(), vec_colidx.end(), colidx);
    for (int i = 0; i < (*nRow); i++)
    {
        for (int j = rowptr[i]; j < rowptr[i + 1]; j++)
        {
            // printf("%d::%d %d :: %d\n", i, rowptr[i], rowptr[i + 1], j);
            c = colidx[j];
            // printf("%d %d - %d :: %d %d\n", *nRow, *nCol, (*nRow) * (*nCol), i, c);
            (*mat)[i * (*nCol) + c] = __float2half(1.0f);
        }
    }
    vec_colidx.clear();
}
void smtx_input_to_CSR(ifstream &fin, CSR *csr_mat, Arguments *arg)
{
}
