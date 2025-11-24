# Makefile for libortho CPU tests

CC = gcc
CFLAGS = -Wall -Wextra -O3 -g -std=c11 -D_POSIX_C_SOURCE=200809L -ffast-math -funroll-loops
INCLUDES = -I../include
LDFLAGS = -lm

# Source files
SRC_DIR = ..
TEST_SRC = test_cpu_forward.c
BENCHMARK_SRC = benchmark_null_test.c
ORTHO_SRC = $(SRC_DIR)/src/ortho.c
CUDA_TEST_SRC = test_cuda_kernel.cu
TENSOR_CORE_TEST_SRC = test_tensor_core.cu

# Output
TEST_BIN = test_cpu_forward
BENCHMARK_BIN = benchmark_null_test
CUDA_TEST_BIN = test_cuda_kernel
TENSOR_CORE_TEST_BIN = test_tensor_core

NVCC = nvcc
CUDA_CFLAGS = -O3 -arch=sm_75 -arch=sm_80 -arch=sm_86 --expt-relaxed-constexpr

.PHONY: all clean test benchmark cuda-test tensor-core-test check-gpu

all: $(TEST_BIN) $(BENCHMARK_BIN)

$(TEST_BIN): $(TEST_SRC) $(ORTHO_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TEST_BIN) $(TEST_SRC) $(ORTHO_SRC) $(LDFLAGS)

$(BENCHMARK_BIN): $(BENCHMARK_SRC) $(ORTHO_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(BENCHMARK_BIN) $(BENCHMARK_SRC) $(ORTHO_SRC) $(LDFLAGS)

$(CUDA_TEST_BIN): $(CUDA_TEST_SRC)
	$(NVCC) $(CUDA_CFLAGS) -o $(CUDA_TEST_BIN) $(CUDA_TEST_SRC) -lcudart

$(TENSOR_CORE_TEST_BIN): $(TENSOR_CORE_TEST_SRC)
	$(NVCC) $(CUDA_CFLAGS) -I../include -o $(TENSOR_CORE_TEST_BIN) $(TENSOR_CORE_TEST_SRC) ../src/dual_gemm_tensor_core.cu -lcudart

test: $(TEST_BIN)
	./$(TEST_BIN)

benchmark: $(BENCHMARK_BIN)
	./$(BENCHMARK_BIN)

cuda-test: $(CUDA_TEST_BIN)
	./$(CUDA_TEST_BIN)

tensor-core-test: $(TENSOR_CORE_TEST_BIN)
	./$(TENSOR_CORE_TEST_BIN)

check-gpu:
	python3 check_gpu.py

clean:
	rm -f $(TEST_BIN) $(BENCHMARK_BIN) $(CUDA_TEST_BIN) $(TENSOR_CORE_TEST_BIN)

