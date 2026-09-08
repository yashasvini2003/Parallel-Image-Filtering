#!/usr/bin/env bash
set -euo pipefail

MAX_SIGMA="${1:?Usage: $0 <max_sigma> [output_file] [input_images...]}"
OUTPUT="${2:-benchmark.csv}"
IMAGES=("${@:3}")
if [ ${#IMAGES[@]} -eq 0 ]; then
  IMAGES=(img/input_500.jpg img/input_1000.jpg img/input_2000.jpg img/input_3000.jpg)
fi

BINDIR="$(dirname "$0")/bin"
PROGRAMS=(linear naive_iterative separable_iterative separable_recursive_deriche separable_recursive_van_vliet)
RUNS=4
WARMUP=1

{
  echo "sigma,image_size,program,time_ms"

  for img in "${IMAGES[@]}"; do
    base=$(basename "$img" .jpg)
    size=${base##*_}
    echo "=== ${size}x${size} ===" >&2

    for sigma in $(seq 1 "$MAX_SIGMA"); do
      for prog in "${PROGRAMS[@]}"; do
        times=()
        for run in $(seq 1 "$RUNS"); do
          t="$("$BINDIR/$prog" --sigma "$sigma" --input "$img" 2>/dev/null | sed -n 's/Processing time: \([0-9.]*\) ms/\1/p')"
          if [ -z "$t" ]; then
            t="NaN"
          fi
          times+=("$t")
        done
        sum=0
        count=0
        for i in $(seq "$WARMUP" "$((RUNS - 1))"); do
          val="${times[$i]}"
          if [ "$val" != "NaN" ]; then
            sum="$(awk "BEGIN {print $sum + $val}")"
            count=$((count + 1))
          fi
        done
        if [ "$count" -gt 0 ]; then
          awk -v s="$sigma" -v z="$size" -v p="$prog" -v sum="$sum" -v cnt="$count" \
            'BEGIN {printf "%d,%d,%s,%.6f\n", s, z, p, sum / cnt}'
        else
          echo "$sigma,$size,$prog,NaN"
        fi
      done
    done
  done
} > "$OUTPUT"

echo "Saved to $OUTPUT"
