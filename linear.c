#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#include "lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "lib/stb_image_write.h"

static float *kernel = NULL;
static int kernel_size = 0;
static float sigma_val = 10.0f;

static int compute_kernel_size(float sigma) {
  int k = (int)(6.0f * sigma + 1.0f);
  if (k % 2 == 0)
    k++;
  return k;
}

static void compute_kernel(void) {
  float sum = 0.0f;
  int half = kernel_size / 2;
  float s2 = 2.0f * sigma_val * sigma_val;
  for (int y = -half; y <= half; y++) {
    for (int x = -half; x <= half; x++) {
      float v = expf(-(float)(x * x + y * y) / s2);
      kernel[(y + half) * kernel_size + (x + half)] = v;
      sum += v;
    }
  }
  for (int i = 0; i < kernel_size * kernel_size; i++)
    kernel[i] /= sum;
}

int main(int argc, char **argv) {
  float sigma = 10.0f;
  const char *input_path = "img/input.jpg";
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--sigma") == 0 || strcmp(argv[i], "-s") == 0) {
      if (i + 1 < argc)
        sigma = atof(argv[++i]);
    } else if (strcmp(argv[i], "--input") == 0 || strcmp(argv[i], "-i") == 0) {
      if (i + 1 < argc)
        input_path = argv[++i];
    }
  }

  int w, h, channels;
  unsigned char *img = stbi_load(input_path, &w, &h, &channels, 3);
  if (!img) {
    fprintf(stderr, "Failed to load %s\n", input_path);
    return 1;
  }

  kernel_size = compute_kernel_size(sigma);
  sigma_val = sigma;
  kernel = malloc((size_t)kernel_size * kernel_size * sizeof(float));
  if (!kernel) {
    fprintf(stderr, "Failed to allocate kernel\n");
    stbi_image_free(img);
    return 1;
  }
  compute_kernel();

  unsigned char *out = malloc((size_t)w * h * 3);
  int half = kernel_size / 2;

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      float r = 0, g = 0, b = 0;
      for (int ky = 0; ky < kernel_size; ky++) {
        for (int kx = 0; kx < kernel_size; kx++) {
          int px = x + kx - half;
          int py = y + ky - half;
          if (px < 0)
            px = 0;
          if (px >= w)
            px = w - 1;
          if (py < 0)
            py = 0;
          if (py >= h)
            py = h - 1;
          float k = kernel[ky * kernel_size + kx];
          int idx = (py * w + px) * 3;
          r += img[idx + 0] * k;
          g += img[idx + 1] * k;
          b += img[idx + 2] * k;
        }
      }
      int oidx = (y * w + x) * 3;
      out[oidx + 0] = (unsigned char)(r + 0.5f);
      out[oidx + 1] = (unsigned char)(g + 0.5f);
      out[oidx + 2] = (unsigned char)(b + 0.5f);
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double ms =
      (t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

  char output_path[1024] = "img/output.jpg";
  const char *input_base = strrchr(input_path, '/');
  input_base = input_base ? input_base + 1 : input_path;
  if (strncmp(input_base, "input", 5) == 0) {
    snprintf(output_path, sizeof(output_path), "img/output%s", input_base + 5);
  }
  stbi_write_jpg(output_path, w, h, 3, out, 95);
  stbi_image_free(img);
  free(out);
  free(kernel);
  printf("Processing time: %.3f ms\n", ms);
  return 0;
}
