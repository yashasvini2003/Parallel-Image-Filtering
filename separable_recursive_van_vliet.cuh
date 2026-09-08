/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _RECURSIVEGAUSSIAN_KERNEL_CU_
#define _RECURSIVEGAUSSIAN_KERNEL_CU_

#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace cg = cooperative_groups;

#include "lib/helper_cuda.h"
#include "lib/helper_math.h"

#define BLOCK_DIM 16
#define CLAMP_TO_EDGE 1

// Transpose kernel (see transpose CUDA Sample for details)
__global__ void d_transpose(uint *odata, uint *idata, int width, int height) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  __shared__ uint block[BLOCK_DIM][BLOCK_DIM + 1];

  // read the matrix tile into shared memory
  unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
  unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;

  if ((xIndex < width) && (yIndex < height)) {
    unsigned int index_in = yIndex * width + xIndex;
    block[threadIdx.y][threadIdx.x] = idata[index_in];
  }

  cg::sync(cta);

  // write the transposed matrix tile to global memory
  xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
  yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;

  if ((xIndex < height) && (yIndex < width)) {
    unsigned int index_out = yIndex * height + xIndex;
    odata[index_out] = block[threadIdx.x][threadIdx.y];
  }
}

// RGBA version
// reads from 32-bit uint array holding 8-bit RGBA

// convert floating point rgba color to 32-bit integer
__device__ uint rgbaFloatToInt(float4 rgba) {
  rgba.x = __saturatef(rgba.x); // clamp to [0.0, 1.0]
  rgba.y = __saturatef(rgba.y);
  rgba.z = __saturatef(rgba.z);
  rgba.w = __saturatef(rgba.w);
  return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) |
         (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

// convert from 32-bit int to float4
__device__ float4 rgbaIntToFloat(uint c) {
  float4 rgba;
  rgba.x = (c & 0xff) / 255.0f;
  rgba.y = ((c >> 8) & 0xff) / 255.0f;
  rgba.z = ((c >> 16) & 0xff) / 255.0f;
  rgba.w = ((c >> 24) & 0xff) / 255.0f;
  return rgba;
}

/*
  Van Vliet 3rd-order Recursive Gaussian Filter Kernel
*/

__global__ void d_gaussian_rgba(uint *id, uint *od, int w, int h,
                                float a0, float b1, float b2, float b3,
                                float coefp, float coefn) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= w)
    return;

  id += x;
  od += x;

  // ----------------------------
  // Forward pass (Causal)
  // ----------------------------
  float4 yp1 = make_float4(0.0f);
  float4 yp2 = make_float4(0.0f);
  float4 yp3 = make_float4(0.0f);

#if CLAMP_TO_EDGE
  float4 init_p = rgbaIntToFloat(*id);
  yp1 = yp2 = yp3 = init_p * coefp;
#endif

  for (int y = 0; y < h; y++) {
    float4 xc = rgbaIntToFloat(*id);

    // Van Vliet forward 3rd-order differential formula
    float4 yc = a0 * xc + b1 * yp1 + b2 * yp2 + b3 * yp3;

    *od = rgbaFloatToInt(yc);

    id += w;
    od += w;

    // Shift history windows correctly
    yp3 = yp2;
    yp2 = yp1;
    yp1 = yc;
  }

  // Rewind od pointer to point to the last element in the column
  od -= w;

  // ----------------------------
  // Backward pass (Anti-causal)
  // ----------------------------
  float4 yn1 = make_float4(0.0f);
  float4 yn2 = make_float4(0.0f);
  float4 yn3 = make_float4(0.0f);

#if CLAMP_TO_EDGE
  // Van Vliet reads the forward pass output (*od) as input for the backward pass
  float4 init_n = rgbaIntToFloat(*od);
  yn1 = yn2 = yn3 = init_n * coefn;
#endif

  for (int y = h - 1; y >= 0; y--) {
    // Read from the forward pass output stored in od
    float4 xc = rgbaIntToFloat(*od);

    // Van Vliet backward 3rd-order differential formula
    float4 yc = a0 * xc + b1 * yn1 + b2 * yn2 + b3 * yn3;

    // Shift history windows correctly
    yn3 = yn2;
    yn2 = yn1;
    yn1 = yc;

    // Overwrite with the cascading result instead of adding it
    *od = rgbaFloatToInt(yc);

    od -= w; // move to previous row
  }
}

#endif // #ifndef _GAUSSIAN_KERNEL_H_
