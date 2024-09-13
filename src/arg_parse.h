#ifndef _ARG_PARSE_CUH
#define _ARG_PARSE_CUH
#include "general.cuh"
#include <argp.h>
#include <unistd.h>
static const char *dtypes_string[] = {
    "FP16",
    "FP32",
    "FP64"};

static const char *unit_mma_string[] = {
    "M16N8K8 - A,B: Fp16, C : FP32",
    "M16N8K16 - A, B: FP16, C: FP32",
    "M8N8K4 - A, B: FP64, C:FP64"};
class Arguments
{
public:
    int M;
    int N;
    int K;
    int num_iter;
    bool pattern_only;
    bool input_is_zero_based;
    bool input_file_exists;
    float target_sparsity;
    float tolerance;
    entry_dtype a_type, b_type, c_type;
    char *input_fname;
    int ell_block_size;

    Arguments(int argc, char *argv[])
    {
        M = DEFAULT_M;
        N = DEFAULT_N;
        K = DEFAULT_K;
        num_iter = N_ITER;
        target_sparsity = TARGET_SPARSITY;
        tolerance = DEFAULT_TOL;
        a_type = fp16;
        b_type = fp16;
        c_type = fp32;
        input_fname = NULL;
        pattern_only = false;
        input_is_zero_based = true;
        input_file_exists = false;
        ell_block_size = 32;
        ParseArgs(argc, argv);
        print();
    }
    ~Arguments()
    {
    }
    void ParseArgs(int argc, char *argv[])
    {
        char c_opt;
        while ((c_opt = getopt(argc, argv, "m:n:k:i:t:s:f:d:p:b:e:")) != -1)
        {
            switch (c_opt)
            {
            case 'm':
                M = std::stoi(optarg);
                break;
            case 'n':
                N = std::stoi(optarg);
                break;
            case 'k':
                K = std::stoi(optarg);
                break;
            case 'i':
                num_iter = std::stoi(optarg);
                break;
            case 't':
                tolerance = std::stof(optarg);
                break;
            case 's':
                target_sparsity = std::stof(optarg);
                break;
            case 'f':
                input_fname = (char *)malloc(strlen(optarg) * sizeof(char));
                strcpy(input_fname, optarg);
                break;
            case 'p':
                // pattern only true
                pattern_only = (std::stoi(optarg) == 1);
                break;
            case 'b':
                // zero base or one base
                input_is_zero_based = (std::stoi(optarg) == 0);
                break;
            case 'e':
                // ell block size
                ell_block_size = (std::stoi(optarg));
                break;
            case 'd':
                entry_dtype dt = (entry_dtype)std::stoi(optarg);
                if (dt == fp16)
                {
                    a_type = fp16;
                    b_type = fp16;
                    c_type = fp32;
                }
                else if (dt == fp64)
                {
                    a_type = b_type = c_type = fp64;
                }
                else
                {
                    fprintf(stderr, "Not Implemented error for the specified dtype\n");
                    exit(EXIT_FAILURE);
                }
                break;
            }
        }
        if (input_fname != NULL && access(input_fname, F_OK) != 0)
        {
            fprintf(stderr, "Input file not exists error\n");
            exit(EXIT_FAILURE);
        }
        else if (input_fname != NULL)
        {
            input_file_exists = true;
            // input_fstream.open();
        }
        else
        {
            input_file_exists = false;
        }
    }

    void print()
    {
#if VERBOSE == true
        printf("===================== Arguments ==========================\n");

        printf("Tolerance : %f\n", tolerance);
        printf("A type : %s, B type : %s, C type : %s\n", dtypes_string[a_type], dtypes_string[b_type], dtypes_string[c_type]);
        printf("N iter : %d\n", num_iter);
        printf("Ell block size : %d \n", ell_block_size);
        if (input_fname == NULL)
        {
            printf("No input file have been passed ... generate synthetic with random entries\n");
            printf("The matrix size .. M: %d , N: %d, K: %d\n", M, N, K);
            printf("Target Sparsity : %f\n", target_sparsity);
        }
        else
        {
            printf("Input file name : %s\n", input_fname);
        }

        if (pattern_only)
        {
            printf("Matrix is specified as pattern only matrix\n");
        }
        if (!input_is_zero_based)
        {
            printf("Input matrix is specified as one based index matrix\n");
        }
        printf("================== End of Arguments ======================\n");
#endif
    }
};
#endif