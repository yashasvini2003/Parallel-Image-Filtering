# Parallel Image Filtering with CUDA

[![CUDA](https://img.shields.io/badge/CUDA-GPU%20Acceleration-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![C](https://img.shields.io/badge/C-CPU%20Baseline-A8B9CC?logo=c&logoColor=black)](https://www.c-language.org/)
[![Python](https://img.shields.io/badge/Python-Benchmark%20Visualization-3776AB?logo=python&logoColor=white)](https://www.python.org/)

A performance-focused implementation of **Gaussian image filtering** that compares a sequential CPU baseline with four GPU-accelerated CUDA approaches. The project explores how algorithm design, memory movement, separable convolution, and recursive filtering affect execution time as image size and Gaussian standard deviation (`sigma`) increase.

## Project goals

- Implement Gaussian blur using both CPU and GPU computation.
- Compare direct convolution, separable convolution, and recursive IIR approximations.
- Measure scalability across multiple image resolutions and sigma values.
- Calculate GPU speedup relative to a sequential CPU baseline.
- Visualize timing, speedup, image-size scaling, and output quality.

## Implementations

| Implementation | Device | Approach | Time complexity |
|---|---|---|---|
| `linear.c` | CPU | Full 2D Gaussian convolution used as the sequential reference | `O(Nσ²)` |
| `naive_iterative.cu` | GPU | Direct 2D convolution; each CUDA thread processes one output pixel | `O(Nσ²)` |
| `separable_iterative.cu` | GPU | Two 1D convolution passes with matrix transposition | `O(Nσ)` |
| `separable_recursive_deriche.cu` | GPU | Deriche recursive IIR approximation with forward and backward passes | `O(N)` |
| `separable_recursive_van_vliet.cu` | GPU | Third-order Van Vliet recursive IIR approximation | `O(N)` |

Here, `N = width × height`, and the direct Gaussian kernel dimension is approximately `6σ + 1`.

## How it works

1. An RGB image is loaded using `stb_image`.
2. The CPU implementation performs direct two-dimensional convolution.
3. CUDA implementations copy packed RGBA pixels to GPU memory.
4. GPU kernels execute direct, separable, or recursive filtering.
5. Separable implementations transpose intermediate image data so the same vertical-pass strategy can be reused horizontally.
6. CUDA events measure GPU processing time; the CPU baseline uses a monotonic system clock.
7. The filtered image is copied back to host memory and written as a JPEG.
8. Benchmark scripts aggregate repeated runs into CSV data for analysis and visualization.

## Technologies

- **Languages:** C, CUDA C/C++, Python, Bash
- **Parallel computing:** NVIDIA CUDA, GPU kernels, thread blocks, device memory, host-device transfers
- **Algorithms:** Gaussian convolution, separable filtering, Deriche IIR filtering, Van Vliet IIR filtering, matrix transposition
- **Build and automation:** GNU Make, shell scripting
- **Image processing:** `stb_image`, `stb_image_write`
- **Data analysis:** pandas, NumPy, Matplotlib, Pillow

## Repository structure

```text
.
├── linear.c                               # Sequential CPU reference
├── naive_iterative.cu/.cuh               # Direct 2D CUDA convolution
├── separable_iterative.cu/.cuh           # Two-pass separable CUDA filter
├── separable_recursive_deriche.cu/.cuh   # Deriche recursive CUDA filter
├── separable_recursive_van_vliet.cu/.cuh # Van Vliet recursive CUDA filter
├── Makefile                               # Build and run targets
├── benchmark.sh                           # Repeated benchmark runner
├── benchmark.csv                          # Recorded timing data
├── plot_benchmarks.py                     # Timing and speedup charts
├── plot_image_demo.py                     # Visual output comparisons
├── DOCUMENTATION.md                       # Algorithm notes
└── LICENSE                                # MIT License
```

The Makefile and plotting scripts also expect these folders:

```text
lib/
├── stb_image.h
├── stb_image_write.h
├── helper_string.h
├── helper_cuda.h
└── helper_math.h

img/
├── input_500.jpg
├── input_1000.jpg
├── input_2000.jpg
└── input_3000.jpg
```

> **Important:** Ensure the `lib/` headers and `img/` input files are present before building or running the project.

## Requirements

### Build requirements

- A Unix-like environment
- NVIDIA GPU with CUDA support
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) and `nvcc`
- `clang`
- GNU Make
- Standard C math library

### Visualization requirements

```bash
python3 -m pip install pandas numpy matplotlib pillow
```

## Build

Clone the repository and compile all CPU and GPU implementations:

```bash
git clone https://github.com/yashasvini2003/Parallel-Image-Filtering.git
cd Parallel-Image-Filtering
make
```

Compiled executables are created in `bin/`.

## Run

Run every implementation using the default input and `sigma = 10`:

```bash
make run
```

Override the sigma value or input image:

```bash
make run SIGMA=15 INPUT=img/input_3000.jpg
```

Run only the CPU or GPU implementations:

```bash
make run_cpu SIGMA=10 INPUT=img/input_1000.jpg
make run_gpu SIGMA=10 INPUT=img/input_1000.jpg
```

You can also run a compiled executable directly:

```bash
./bin/separable_iterative --sigma 10 --input img/input_3000.jpg
```

Each program prints its processing time in milliseconds and writes a filtered JPEG under `img/`. When multiple implementations use the same input name, later runs may overwrite the preceding output file.

## Benchmarking

Run all five implementations for sigma values from 1 through a chosen maximum:

```bash
./benchmark.sh 30 benchmark.csv
```

By default, the script benchmarks 500×500, 1000×1000, 2000×2000, and 3000×3000 images. Each configuration includes one warm-up followed by three measured runs, whose mean is written to the CSV file.

Custom image paths can be supplied after the output filename:

```bash
./benchmark.sh 20 results.csv img/input_500.jpg img/input_2000.jpg
```

## Recorded benchmark example

The committed `benchmark.csv` contains 600 measurements covering five implementations, four image sizes, and sigma values from 1 to 30. For a 3000×3000 image at `sigma = 30`, it records:

| Implementation | Processing time |
|---|---:|
| Linear CPU | 322,670.333 ms |
| Naïve iterative GPU | 4,483.667 ms |
| Separable iterative GPU | 438.363 ms |
| Recursive Deriche GPU | 11.892 ms |
| Recursive Van Vliet GPU | 9.941 ms |

These figures illustrate the scaling advantage of recursive GPU filters at large sigma values. Results are hardware- and environment-dependent and should be reproduced on the target system before drawing general performance conclusions.

## Generate visualizations

Create benchmark charts:

```bash
python3 plot_benchmarks.py
```

The script generates:

- `img/time_vs_sigma.png`
- `img/speedup_vs_sigma.png`
- `img/time_vs_imagesize.png`

After generating the expected algorithm output images, create visual comparisons:

```bash
python3 plot_image_demo.py
```

This produces sigma-progression and algorithm-comparison figures under `img/`.

## Key observations

- Direct 2D convolution becomes increasingly expensive as sigma enlarges the kernel.
- A naïve CUDA port retains the same asymptotic work but parallelizes pixel processing.
- Separable convolution reduces a 2D filter to two 1D passes.
- Recursive Deriche and Van Vliet approximations perform constant work per pixel with respect to sigma.
- Faster execution introduces engineering trade-offs involving approximation accuracy, boundary handling, synchronization, memory transfers, and GPU hardware availability.

## Limitations and future improvements

- The CUDA implementations require compatible NVIDIA hardware.
- Benchmark timings focus on the filtering stage and are not a complete end-to-end application latency measurement.
- Recursive approaches approximate the Gaussian and should be evaluated with quantitative image-quality metrics.
- Output naming can overwrite results when several implementations process the same image.
- Future work could add automated correctness tests, PSNR/SSIM comparisons, CUDA profiling, shared-memory optimizations, additional image formats, and CI-based CPU build checks.

## Author

**Yashasvini Bhanuraj**

- [GitHub](https://github.com/yashasvini2003)
- [LinkedIn](https://www.linkedin.com/in/yashasvini-bhanuraj-0a7a13202/)
