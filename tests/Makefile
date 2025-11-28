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

# Detect CUDA version and set architectures accordingly
# sm_100 (Blackwell) requires CUDA 12.8+
# Try to detect CUDA version, fallback to base architectures if detection fails
CUDA_VERSION_STR := $(shell nvcc --version 2>/dev/null | grep -o 'release [0-9]\+\.[0-9]\+' | head -1 | sed 's/release //' || echo "0.0")
CUDA_MAJOR := $(shell echo $(CUDA_VERSION_STR) | cut -d. -f1)
CUDA_MINOR := $(shell echo $(CUDA_VERSION_STR) | cut -d. -f2)

# Base architectures (supported by CUDA 11.0+)
CUDA_ARCHS := -arch=sm_75 -arch=sm_80 -arch=sm_86 -arch=sm_89

# Add sm_100 only if CUDA >= 12.8
# Use numeric comparison: if major > 12 OR (major == 12 AND minor >= 8)
CUDA_SUPPORTS_SM100 := $(shell [ $(CUDA_MAJOR) -gt 12 ] 2>/dev/null || ([ $(CUDA_MAJOR) -eq 12 ] 2>/dev/null && [ $(CUDA_MINOR) -ge 8 ] 2>/dev/null) && echo "yes" || echo "no")
ifeq ($(CUDA_SUPPORTS_SM100),yes)
    CUDA_ARCHS += -arch=sm_100
    $(info CUDA $(CUDA_VERSION_STR) detected: Including sm_100 (Blackwell) support)
else
    $(info CUDA $(CUDA_VERSION_STR) detected: sm_100 not supported (requires CUDA 12.8+))
endif

# FIXED: Use -gencode instead of -arch for better compatibility
# This ensures kernel images are generated for each architecture
CUDA_CFLAGS = -O3 --expt-relaxed-constexpr
# Generate code for each architecture explicitly
CUDA_CFLAGS += -gencode arch=compute_75,code=sm_75
CUDA_CFLAGS += -gencode arch=compute_80,code=sm_80
CUDA_CFLAGS += -gencode arch=compute_86,code=sm_86
CUDA_CFLAGS += -gencode arch=compute_89,code=sm_89
# Add sm_100 only if CUDA >= 12.8
ifeq ($(CUDA_SUPPORTS_SM100),yes)
    CUDA_CFLAGS += -gencode arch=compute_100,code=sm_100
endif
# sm_75: Turing (RTX 20xx) - CUDA 10.0+
# sm_80: Ampere (A100) - CUDA 11.0+
# sm_86: Ampere consumer (RTX 30xx) - CUDA 11.0+
# sm_89: Ada Lovelace (GTX 4050, RTX 40xx) - CUDA 11.8+
# sm_100: Blackwell (RTX 5060, RTX 50 series) - CUDA 12.8+

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

