# Makefile for libortho CPU tests

CC = gcc
CFLAGS = -Wall -Wextra -O2 -g -std=c11
INCLUDES = -I../include
LDFLAGS = -lm

# Source files
SRC_DIR = ..
TEST_SRC = test_cpu_forward.c
ORTHO_SRC = $(SRC_DIR)/src/ortho.c

# Output
TEST_BIN = test_cpu_forward

.PHONY: all clean test

all: $(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(ORTHO_SRC)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TEST_BIN) $(TEST_SRC) $(ORTHO_SRC) $(LDFLAGS)

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f $(TEST_BIN)

