LD = -Xcompiler=-fopenmp -lcublas -lcusparse -lcusparseLt -lcudart 
NVFLAG = -arch=sm_86

test: test.o miscalleneous.o gemm_cublas.o spmm_cusparse.o  our_spmm.o file_input.o
	nvcc -g -o test $(INCLUDE) $(LD) $(NVFLAG)  test.o miscalleneous.o spmm_cusparse.o  our_spmm.o file_input.o gemm_cublas.o



%.o : %.cu
	nvcc $(INCLUDE) $(LD) $(NVFLAG) $(DEFS) -o $@ -c $< 


clean:
	rm *.o
	rm test
