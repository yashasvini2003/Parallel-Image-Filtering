#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

#include "separable_iterative.cuh"

#define BLOCK_DIM 16

static inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

static void transpose(uint32_t *src, uint32_t *dst, int width, int height) {
  dim3 block(BLOCK_DIM, BLOCK_DIM);
  dim3 grid(iDivUp(width, BLOCK_DIM), iDivUp(height, BLOCK_DIM));

  d_transpose<<<grid, block>>>((uint *)dst, (uint *)src, width, height);
  getLastCudaError("d_transpose launch failed");
}

static void gaussianFilterRGBA(uint32_t *d_src, uint32_t *d_dst,
                               uint32_t *d_tmp, int width, int height,
                               float sigma) {
  int threads = 256;

  // Pass 1: vertical (1D)
  d_gaussian_rgba<<<iDivUp(width, threads), threads>>>(
      (uint *)d_src, (uint *)d_tmp, width, height, sigma);
  getLastCudaError("d_gaussian1D_separable_rgba vertical pass failed");

  // transpose
  transpose(d_tmp, d_dst, width, height);

  // Pass 2: vertical again (acts as horizontal)
  d_gaussian_rgba<<<iDivUp(height, threads), threads>>>(
      (uint *)d_dst, (uint *)d_tmp, height, width, sigma);
  getLastCudaError("d_gaussian1D_separable_rgba horizontal pass failed");

  // transpose back
  transpose(d_tmp, d_dst, height, width);
}

int main(int argc, char **argv) {
  float sigma = 10.0f;
  const char *input_path = "img/input.jpg";
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--sigma") == 0 || strcmp(argv[i], "-s") == 0) {
      if (i + 1 < argc) sigma = atof(argv[++i]);
    } else if (strcmp(argv[i], "--input") == 0 || strcmp(argv[i], "-i") == 0) {
      if (i + 1 < argc) input_path = argv[++i];
    }
  }

  int w, h, channels;

  unsigned char *img = stbi_load(input_path, &w, &h, &channels, 3);

  if (!img) {
    fprintf(stderr, "Failed to load %s\n", input_path);
    return 1;
  }

  size_t pixels = (size_t)w * h;

  uint32_t *hostRGBA = (uint32_t *)malloc(pixels * sizeof(uint32_t));

  for (size_t i = 0; i < pixels; i++) {
    uint8_t r = img[i * 3 + 0];
    uint8_t g = img[i * 3 + 1];
    uint8_t b = img[i * 3 + 2];
    uint8_t a = 255;

    hostRGBA[i] = ((uint32_t)a << 24) | ((uint32_t)b << 16) |
                  ((uint32_t)g << 8) | ((uint32_t)r);
  }

  uint32_t *d_src;
  uint32_t *d_dst;
  uint32_t *d_tmp;

  checkCudaErrors(cudaMalloc(&d_src, pixels * sizeof(uint32_t)));
  checkCudaErrors(cudaMalloc(&d_dst, pixels * sizeof(uint32_t)));
  checkCudaErrors(cudaMalloc(&d_tmp, pixels * sizeof(uint32_t)));

  checkCudaErrors(cudaMemcpy(d_src, hostRGBA, pixels * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaEventRecord(start));

  gaussianFilterRGBA(d_src, d_dst, d_tmp, w, h, sigma);

  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));

  float ms = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));

  checkCudaErrors(cudaMemcpy(hostRGBA, d_dst, pixels * sizeof(uint32_t),
                             cudaMemcpyDeviceToHost));

  unsigned char *out = (unsigned char *)malloc(pixels * 3);

  for (size_t i = 0; i < pixels; i++) {
    uint32_t p = hostRGBA[i];

    out[i * 3 + 0] = p & 0xff;
    out[i * 3 + 1] = (p >> 8) & 0xff;
    out[i * 3 + 2] = (p >> 16) & 0xff;
  }

  char output_path[1024] = "img/output.jpg";
  const char *input_base = strrchr(input_path, '/');
  input_base = input_base ? input_base + 1 : input_path;
  if (strncmp(input_base, "input", 5) == 0) {
    snprintf(output_path, sizeof(output_path), "img/output%s", input_base + 5);
  }
  stbi_write_jpg(output_path, w, h, 3, out, 95);

  printf("Processing time: %.3f ms\n", ms);

  checkCudaErrors(cudaFree(d_src));
  checkCudaErrors(cudaFree(d_dst));
  checkCudaErrors(cudaFree(d_tmp));

  free(out);
  free(hostRGBA);
  stbi_image_free(img);

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return 0;
}
