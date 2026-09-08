CC = clang
NVCC = nvcc

BINDIR = bin
LIBS = lib/stb_image.h lib/stb_image_write.h lib/helper_string.h lib/helper_cuda.h lib/helper_math.h

all: $(BINDIR)/linear $(BINDIR)/naive_iterative $(BINDIR)/separable_recursive_deriche $(BINDIR)/separable_recursive_van_vliet $(BINDIR)/separable_iterative

$(BINDIR):
	mkdir -p $(BINDIR)

$(BINDIR)/linear: linear.c $(LIBS) | $(BINDIR)
	$(CC) linear.c -o $@ -O3 -lm

$(BINDIR)/naive_iterative: naive_iterative.cu naive_iterative.cuh $(LIBS) | $(BINDIR)
	$(NVCC) naive_iterative.cu -o $@ -O3

$(BINDIR)/separable_recursive_deriche: separable_recursive_deriche.cu separable_recursive_deriche.cuh $(LIBS) | $(BINDIR)
	$(NVCC) separable_recursive_deriche.cu -o $@ -O3

$(BINDIR)/separable_recursive_van_vliet: separable_recursive_van_vliet.cu separable_recursive_van_vliet.cuh $(LIBS) | $(BINDIR)
	$(NVCC) separable_recursive_van_vliet.cu -o $@ -O3

$(BINDIR)/separable_iterative: separable_iterative.cu separable_iterative.cuh $(LIBS) | $(BINDIR)
	$(NVCC) separable_iterative.cu -o $@ -O3

SIGMA ?= 10.0
INPUT ?= img/input_3000.jpg

run_cpu: $(BINDIR)/linear
	@echo "=== Linear (CPU) ===" && ./$(BINDIR)/linear --sigma $(SIGMA) --input $(INPUT)

run_gpu: $(BINDIR)/naive_iterative $(BINDIR)/separable_iterative $(BINDIR)/separable_recursive_deriche $(BINDIR)/separable_recursive_van_vliet
	@echo "=== Naive Iterative (GPU, full 2D) ===" && \
	 for i in 1 2 3; do ./$(BINDIR)/naive_iterative --sigma $(SIGMA) --input $(INPUT); done && \
	 echo "=== Separable Iterative (GPU, separable 1D) ===" && \
	 for i in 1 2 3; do ./$(BINDIR)/separable_iterative --sigma $(SIGMA) --input $(INPUT); done && \
	 echo "=== Van Vliet (GPU, separable recursive 1D) ===" && \
	 for i in 1 2 3; do ./$(BINDIR)/separable_recursive_van_vliet --sigma $(SIGMA) --input $(INPUT); done && \
	 echo "=== Deriche (GPU, separable recursive 1D) ===" && \
	 for i in 1 2 3; do ./$(BINDIR)/separable_recursive_deriche --sigma $(SIGMA) --input $(INPUT); done

run: run_cpu
	@echo "" && echo "=== GPU ==="
	$(MAKE) run_gpu

clean:
	rm -rf $(BINDIR)

.PHONY: all run run_cpu run_gpu clean
