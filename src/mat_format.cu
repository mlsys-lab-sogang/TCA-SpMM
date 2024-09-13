#include "mat_format.cuh"
template <typename I_t, typename O_t>
void format_conversion(mat_format input_format, mat_format output_format, I_t source, O_t destination)
{
    if (input_format == output_format)
    {
        printf("Invalid format conversion request %s -> %s\n", MATTYPE_STRING[input_format], MATTYPE_STRING[output_format]);
        return;
    }
    printf("Converting %s type matrix into %s type matrix\n", MATTYPE_STRING[input_format], MATTYPE_STRING[output_format]);
    if (input_format == dense)
    {
        if (output_format == coo)
        {
        }
        else if (output_format == csr)
        {
        }
    }
    else if (input_format == coo)
    {
        if (output_format == dense)
        {
        }
        else if (output_format == csr)
        {
        }
    }
    else if (input_format == csr)
    {
        if (output_format == dense)
        {
        }
        else if (output_format == coo)
        {
        }
    }
}