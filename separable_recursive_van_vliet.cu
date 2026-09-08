#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

#include "separable_recursive_van_vliet.cuh"

#define BLOCK_DIM 16

static inline int iDivUp(int a, int b) { return (a + b - 1) / b; }

static void transpose(uint32_t *src, uint32_t *dst, int width, int height) {
  dim3 block(BLOCK_DIM, BLOCK_DIM);
  dim3 grid(iDivUp(width, BLOCK_DIM), iDivUp(height, BLOCK_DIM));

  d_transpose<<<grid, block>>>((uint *)dst, (uint *)src, width, height);
  getLastCudaError("d_transpose launch failed");
}

static void gaussianFilterRGBA(uint32_t *d_src, uint32_t *d_dst,
                               uint32_t *d_temp, int width, int height,
                               float sigma) {
  const float nsigma = sigma < 0.1f ? 0.1f : sigma;

  // Real Authentic 3rd-order Van Vliet Pole Scaling Parameters
  float q = 0.0f;
  if (nsigma >= 2.5f) {
    q = 0.98711f * nsigma - 0.96330f;
  } else {
    q = 3.97156f - 4.14554f * sqrtf(1.0f - 0.26891f * nsigma);
  }

  const float q2 = q * q;
  const float q3 = q2 * q;

  // Mathematical constants calculated according to Van Vliet's equations
  const float b0 = 1.57825f + 2.44413f * q + 1.4281f * q2 + 0.422205f * q3;
  const float b1 = (2.44413f * q + 2.85619f * q2 + 1.26661f * q3) / b0;
  const float b2 = -(1.4281f * q2 + 1.26661f * q3) / b0;
  const float b3 = (0.422205f * q3) / b0;
  
  // Normalization factor
  const float a0 = 1.0f - (b1 + b2 + b3);

  // Border steady state tracking gains
  const float coefp = a0 / (1.0f - b1 - b2 - b3);
  const float coefn = coefp; 

  int threads = 256;

  // Vertical Pass
  d_gaussian_rgba<<<iDivUp(width, threads), threads>>>(
      (uint *)d_src, (uint *)d_temp, width, height, a0, b1, b2, b3, coefp, coefn);
  getLastCudaError("d_recursiveGaussian_rgba vertical pass failed");

  transpose(d_temp, d_dst, width, height);

  // Horizontal Pass (run vertically on transposed matrix data)
  d_gaussian_rgba<<<iDivUp(height, threads), threads>>>(
      (uint *)d_dst, (uint *)d_temp, height, width, a0, b1, b2, b3, coefp, coefn);
  getLastCudaError("d_recursiveGaussian_rgba horizontal pass failed");

  transpose(d_temp, d_dst, height, width);
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

  int w;
  int h;
  int channels;

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

    hostRGBA[i] =
        ((uint32_t)a << 24) | ((uint32_t)b << 16) | ((uint32_t)g << 8) | r;
  }

  uint32_t *d_src;
  uint32_t *d_dst;
  uint32_t *d_tmp;

  checkCudaErrors(cudaMalloc(&d_src, pixels * sizeof(uint32_t)));
  checkCudaErrors(cudaMalloc(&d_dst, pixels * sizeof(uint32_t)));
  checkCudaErrors(cudaMalloc(&d_tmp, pixels * sizeof(uint32_t)));

  checkCudaErrors(cudaMemcpy(d_src, hostRGBA, pixels * sizeof(uint32_t),
                             cudaMemcpyHostToDevice));

  cudaEvent_t start;
  cudaEvent_t stop;

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