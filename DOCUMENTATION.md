# Gaussian Blur

CPU and GPU implementations of Gaussian blur on images, comparing
different algorithmic strategies.

## Implementations

| File                            | Device | Description                                                                                          | Per-pixel | Total      |
| ------------------------------- | ------ | ---------------------------------------------------------------------------------------------------- | --------- | ---------- |
| `linear.c`                      | CPU    | Baseline — full 2D convolution kernel                                                                | O(k²)     | **O(Nσ²)** |
| `naive_iterative`               | GPU    | Direct 2D convolution — each thread evaluates the full kernel over its neighborhood                  | O(k²)     | **O(Nσ²)** |
| `separable_iterative`           | GPU    | Separable 1D convolution — two passes (horizontal then vertical) using truncated 1D Gaussian weights | O(k)      | **O(Nσ)**  |
| `separable_recursive_deriche`   | GPU    | Separable recursive IIR filter using Deriche's approximation — forward/backward recurrences          | O(1)      | **O(N)**   |
| `separable_recursive_van_vliet` | GPU    | Separable recursive IIR filter using Van Vliet's approximation — different coefficient structure     | O(1)      | **O(N)**   |

**Note**: `N = W × H` (total pixels in input image), `k = 6σ + 1` (kernel dimension)

## Algorithm trade-offs

- **linear** — the reference. O(k²) per pixel → O(Nσ²) total. Gets expensive quickly at large σ.
- **naive_iterative** — direct port to GPU. Same O(Nσ²) total work, but massively parallel.
- **separable_iterative** — exploits the separability of the Gaussian: two 1D passes instead of one 2D pass. Reduces total complexity from O(Nσ²) to O(Nσ).
- **separable_recursive_deriche / van_vliet** — use IIR (infinite impulse response) filters to approximate the Gaussian. Forward + backward recursion gives O(1) per pixel, O(N) total — completely independent of σ. Deriche and Van Vliet differ in their coefficient derivation and recurrence structure.

## Build & run

```bash
make            # builds all targets into bin/
make run        # runs all on CPU then GPU
SIGMA=15 make run  # override sigma (default 10)
```

## Benchmark

```bash
./benchmark.sh <max_sigma> [output.csv]
```

Runs each program at σ = 1 … max_sigma, averages 4 runs (with 1 warm-up),
and writes timing data to CSV.
