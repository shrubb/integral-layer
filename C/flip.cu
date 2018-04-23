#include <iostream>
#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <device_functions.h>

#include <THC/THC.h>

#define NUM_THREADS 64

template <bool accumulate>
__global__ void flipKernel(const float *input, const int h, const int w, float *output) {

    // go to current row
    input  += blockIdx.x * w;
    output += blockIdx.x * w;

    const int colIdx = blockIdx.y * NUM_THREADS + threadIdx.x;
    
    if (colIdx < w) {
        if (accumulate)
            output[w-1-colIdx] += input[colIdx];
        else
            output[w-1-colIdx]  = input[colIdx];
    }
}

/*
// Kernel that uses shared memory
template <bool accumulate>
__global__ void flipKernel(const float *input, const int h, const int w, float *output) {

    __shared__ float buffer[NUM_THREADS];

    // go to current row
    input  += blockIdx.x * w;
    output += blockIdx.x * w;

    const int colIdx = blockIdx.y * NUM_THREADS + threadIdx.x;
    buffer[NUM_THREADS-1-threadIdx.x] = input[min(colIdx, w-1)];

    __syncthreads();

    const int outputIdx = w - (blockIdx.y + 1) * NUM_THREADS + threadIdx.x;

    if (outputIdx >= 0) {
        if (accumulate)
            output[outputIdx] += buffer[threadIdx.x];
        else
            output[outputIdx]  = buffer[threadIdx.x];
    }
}
*/

extern "C"
void flipCuda(struct THCState *state,
    const float *input, const int h, const int w, float *output, bool accumulate) {

    dim3 numBlocks;
    numBlocks.x = h;
    numBlocks.y = (w + NUM_THREADS - 1) / NUM_THREADS;

    if (accumulate) {
        flipKernel <true>  <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            input, h, w, output);
    } else {
        flipKernel <false> <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            input, h, w, output);
    }
    THCudaCheck(cudaGetLastError());
}
