#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <THC/THC.h>

#include "integral-strided-cuda.hpp"

#define NUM_THREADS 256
#define BLOCK_SIZE 4
#define BLOCK_CHANNELS (NUM_THREADS / (BLOCK_SIZE * BLOCK_SIZE))

using std::max;
using std::min;
using std::floor;
using std::ceil;

// TODO remove this code
#define CUDA_ERROR_CHECK

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

/************************ Integral image computation ************************/

__global__ void accumulateRowsKernel(
    const float * __restrict__ input, float * __restrict__ output,
    const int channels, const int h, const int w);
__global__ void accumulateColsKernel(
    const float * __restrict__ input, float * __restrict__ output,
    const int channels, const int h, const int w);
__global__ void accumulateColsInplaceKernel(
    float * __restrict__ input, const int channels, const int h, const int w);
__global__ void accumulateColsInplaceTransposedKernel(
    float * __restrict__ input, const int channels, const int h, const int w);

extern "C"
void integralImageCuda(THCState *state,
    float *input, float *output, int channels, int h, int w, float *tmp) {
    // input : (channels) x (h) x (w), contiguous
    // output: (channels) x (h+1) x (w+1), contiguous
    // tmp   : at least (channels) * (h+1) * (w+1)

    int blockSize1D, gridSize1D;
    const float ONE = 1.0, ZERO = 0.0;

    cublasSetStream(THCState_getCurrentBlasHandle(state), THCState_getCurrentStream(state));

    int totalCols = channels * w;
    blockSize1D = NUM_THREADS;
    gridSize1D = (totalCols + blockSize1D - 1) / blockSize1D;
    accumulateColsKernel <<<gridSize1D, blockSize1D, 0, THCState_getCurrentStream(state)>>> 
        (input, output, channels, h, w);
    THCudaCheck(cudaGetLastError());

    THCublasCheck(cublasSgeam(
        THCState_getCurrentBlasHandle(state),
        CUBLAS_OP_T, CUBLAS_OP_N, channels * (h+1), w+1,
        &ONE, output, w+1,
        &ZERO, tmp, channels * (h+1),
        tmp, channels * (h+1)));

    int totalRows = channels * h; // actually, number of cols in (w+1) x (channels * (h+1)) image
    blockSize1D = NUM_THREADS;
    gridSize1D = (totalRows + blockSize1D - 1) / blockSize1D;
    accumulateColsInplaceTransposedKernel
        <<<gridSize1D, blockSize1D, 0, THCState_getCurrentStream(state)>>> (tmp, channels, h, w);
    THCudaCheck(cudaGetLastError());

    THCublasCheck(cublasSgeam(
        THCState_getCurrentBlasHandle(state),
        CUBLAS_OP_T, CUBLAS_OP_N, w+1, channels * (h+1),
        &ONE, tmp, channels * (h+1),
        &ZERO, output, w+1,
        output, w+1));
}

__global__ void accumulateRowsKernel(
    const float * __restrict__ input, float * __restrict__ output,
    const int channels, const int h, const int w) {
    // view multichannel image as a multiline single-channel image
    int globalRowIdx = NUM_THREADS * blockIdx.x + threadIdx.x;

    if (globalRowIdx < channels * h) {
        float *outputRow = output + (globalRowIdx + globalRowIdx / h + 1) * (w+1) + 1;
        
        outputRow[-1] = 0;
        double sum = 0;
        for (int i = 0; i < w; ++i) {
            sum += input[globalRowIdx * w + i];
            outputRow[i] = static_cast<float>(sum);
        }

        // need to zero the (0,0) corner of the output separately >:(
        output[(globalRowIdx / h) * (w+1) * (h+1)] = 0;
    }
}

__global__ void accumulateColsKernel(
    const float * __restrict__ input, float * __restrict__ output,
    const int channels, const int h, const int w) {
    // input : (channels * h) x (w)
    // output: (channels * (h+1)) x (w+1) -- first column remains untouched

    // global column index (of total `channels * w` columns in this image):
    const int globalColIdx = NUM_THREADS * blockIdx.x + threadIdx.x;

    if (globalColIdx < channels * w) {
        const int channelIdx = globalColIdx / w;
        const int colIdx = globalColIdx - channelIdx * w;
        
        // jump to the channel of interest:
        int inputPos = channelIdx * h * w + colIdx;
        // (let local columns be 1-indexed: 0-th output column is always zero)
        int outputPos = channelIdx * (h+1) * (w+1) + colIdx + 1;

        output[outputPos] = 0; // 0-th element of every column is always zero
        double sum = 0;
        for (int i = 1; i <= h; ++i) {
            sum += static_cast<double>(input[inputPos + (i-1) * w]);
            output[outputPos + i * (w+1)] = static_cast<float>(sum);
        }
    }
}

__global__ void accumulateColsInplaceTransposedKernel(
    float * __restrict__ input, const int channels, const int h, const int w) {
    // in-place.
    // input: (w+1) x (channels * (h+1))

    // global column index (of total `channels * w` columns in this image):
    const int globalColIdx = NUM_THREADS * blockIdx.x + threadIdx.x;

    if (globalColIdx < channels * h) {
        const int channelIdx = globalColIdx / h;
        // add `channelIdx + 1` to account for one extra column in each horizontally stacked image
        const int colIdx = globalColIdx + channelIdx + 1;

        // need to zero the (0,0) corner of the output separately >:(
        input[channelIdx * (h+1)] = 0;

        input[colIdx] = 0; // first element of every column is always zero
        double sum = 0;
        for (int i = 1; i <= w; ++i) {
            float *currentElement = &input[i * channels * (h+1) + colIdx];
            sum += static_cast<double>(*currentElement);
            *currentElement = static_cast<float>(sum);
        }
    }
}

__global__ void accumulateColsInplaceKernel(
    float * __restrict__ input, const int channels, const int h, const int w) {
    // in-place.
    // input is already a `channels * (h+1) x (w+1)` array

    // global column index (of all `channels * w` columns in this image)
    int colIdx = NUM_THREADS * blockIdx.x + threadIdx.x;

    if (colIdx < channels * w) {
        input += (colIdx / w) * (h+1) * (w+1); // jump to current channel
        colIdx %= w; // switch to local column index,
        ++colIdx;    // it's 1-indexed because first output column is always zero

        input[colIdx] = 0; // first element of every column is always zero
        double sum = 0;
        for (int i = 1; i <= h; ++i) {
            float *currentElement = &input[i * (w+1) + colIdx];
            sum += static_cast<double>(*currentElement);
            *currentElement = static_cast<float>(sum);
        }
    }
}

/************************ updateOutput ************************/

__global__ void forwardNoNormReplicateKernel(
    const float *intData, const int intDataStrideChannel, float *outData,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *const xMin, const float *const xMax,
    const float *const yMin, const float *const yMax) {

    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    outData += id; // outData now points to our output pixel

    const int y = id % w; id /= w;
    const int x = id % h; id /= h;
    const int windowIdx = id % nWindows; id /= nWindows;

    // `id` is now is now the current global input plane number
    intData += id * intDataStrideChannel;

    const int globalWindowIdx = (id % nInputPlane) * nWindows + windowIdx; id /= nInputPlane;
    const int & batchIdx = id;

    if (batchIdx < batchSize) {

        // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
        // `integral()` behavior. Namely, I(x,0) and I(0,y) are
        // always 0 (so it's a C-style array sum).

        // However, when computing sums, we subtract values at points 
        // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
        // and yMin, and thus finally they are not affected.

        const int xMinCurr = (int)ceil(xMin[globalWindowIdx]);
        const int yMinCurr = (int)ceil(yMin[globalWindowIdx]);
        const int xMaxCurr = (int)floor(xMax[globalWindowIdx]) + 1;
        const int yMaxCurr = (int)floor(yMax[globalWindowIdx]) + 1;

        const int t = max(0, min(x+xMinCurr, h-1) );
        const int b = max(1, min(x+xMaxCurr, h)   );
        const int l = max(0, min(y+yMinCurr, w-1) );
        const int r = max(1, min(y+yMaxCurr, w)   );

        float outValue = 0;

        outValue += intData[b*(w+1) + r];
        outValue -= intData[t*(w+1) + r];
        outValue -= intData[b*(w+1) + l];
        outValue += intData[t*(w+1) + l];

        *outData = outValue;
    }
}

__global__ void forwardNoNormReplicateFracKernel(
    const float *intData, const int intDataStrideChannel, float *outData,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *const xMin, const float *const xMax,
    const float *const yMin, const float *const yMax,
    const float *inData, const int inDataStrideRow, const int inDataStrideChannel) {

    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    outData += id; // outData now points to our output pixel

    const int y = id % w; id /= w;
    const int x = id % h; id /= h;
    const int windowIdx = id % nWindows; id /= nWindows;

    // `id` is now is now the current global input plane number
    intData += id * intDataStrideChannel;
    inData  += id *  inDataStrideChannel;

    const int globalWindowIdx = (id % nInputPlane) * nWindows + windowIdx; id /= nInputPlane;
    const int & batchIdx = id;

    if (batchIdx < batchSize) {

        // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
        // `integral()` behavior. Namely, I(x,0) and I(0,y) are
        // always 0 (so it's a C-style array sum).

        // However, when computing sums, we subtract values at points 
        // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
        // and yMin, and thus finally they are not affected.

        const int   xMinCurr = (int)ceil(xMin[globalWindowIdx]);
        const float xMinCurrFrac = (float)xMinCurr - xMin[globalWindowIdx];
        const int   yMinCurr = (int)ceil(yMin[globalWindowIdx]);
        const float yMinCurrFrac = (float)yMinCurr - yMin[globalWindowIdx];

        const float xMaxCurrFrac = xMax[globalWindowIdx] - floor(xMax[globalWindowIdx]);
        const int   xMaxCurr = (int)floor(xMax[globalWindowIdx]) + 1;
        const float yMaxCurrFrac = yMax[globalWindowIdx] - floor(yMax[globalWindowIdx]);
        const int   yMaxCurr = (int)floor(yMax[globalWindowIdx]) + 1;

        const int t = max(0, min(x+xMinCurr, h-1) );
        const int b = max(1, min(x+xMaxCurr, h)   );
        const int l = max(0, min(y+yMinCurr, w-1) );
        const int r = max(1, min(y+yMaxCurr, w)   );

        const int bAdv = max(1, min(x+xMaxCurr+1, h  ));
        const int rAdv = max(1, min(y+yMaxCurr+1, w  ));
        const int tAdv = max(0, min(x+xMinCurr-1, h-1));
        const int lAdv = max(0, min(y+yMinCurr-1, w-1));

        float outValue = 0;

        outValue += intData[b*(w+1) + r];
        outValue -= intData[t*(w+1) + r];
        outValue -= intData[b*(w+1) + l];
        outValue += intData[t*(w+1) + l];

        // -- xMax border
        outValue +=
            ( intData[bAdv*(w+1) + r]
            - intData[b   *(w+1) + r]
            - intData[bAdv*(w+1) + l]
            + intData[b   *(w+1) + l]) * xMaxCurrFrac;

        // -- yMax border
        outValue +=
            ( intData[b*(w+1) + rAdv]
            - intData[b*(w+1) + r   ]
            - intData[t*(w+1) + rAdv]
            + intData[t*(w+1) + r   ]) * yMaxCurrFrac;

        // -- xMin border
        outValue +=
            ( intData[t   *(w+1) + r]
            - intData[tAdv*(w+1) + r]
            - intData[t   *(w+1) + l]
            + intData[tAdv*(w+1) + l]) * xMinCurrFrac;

        // -- yMin border
        outValue +=
            ( intData[b*(w+1) + l   ]
            - intData[b*(w+1) + lAdv]
            - intData[t*(w+1) + l   ]
            + intData[t*(w+1) + lAdv]) * yMinCurrFrac;

        // -- corner pixels
        bool cornerIsValid;

        cornerIsValid = not (
            (x+xMaxCurr >  h-1) |
            (y+yMaxCurr >  w-1) |
            (x+xMaxCurr <=   0) |
            (y+yMaxCurr <=   0));
        outValue += 
            xMaxCurrFrac * yMaxCurrFrac *
            cornerIsValid *
            inData[((x+xMaxCurr) * inDataStrideRow + (y+yMaxCurr)) * cornerIsValid];

        cornerIsValid = not (
            (x+xMinCurr-1 >= h-1) |
            (y+yMaxCurr   >  w-1) |
            (x+xMinCurr-1 <    0) |
            (y+yMaxCurr   <=   0));
        outValue +=
            xMinCurrFrac * yMaxCurrFrac *
            cornerIsValid *
            inData[((x+xMinCurr-1) * inDataStrideRow + (y+yMaxCurr)) * cornerIsValid];

        cornerIsValid = not (
            (x+xMaxCurr   >  h-1) |
            (y+yMinCurr-1 >= w-1) |
            (x+xMaxCurr   <=   0) |
            (y+yMinCurr-1 <    0));
        outValue +=
            xMaxCurrFrac * yMinCurrFrac *
            cornerIsValid *
            inData[((x+xMaxCurr) * inDataStrideRow + (y+yMinCurr-1)) * cornerIsValid];

        cornerIsValid = not (
            (x+xMinCurr-1 >= h-1) |
            (y+yMinCurr-1 >= w-1) |
            (x+xMinCurr-1 <    0) |
            (y+yMinCurr-1 <    0));
        outValue +=
            xMinCurrFrac * yMinCurrFrac *
            cornerIsValid *
            inData[((x+xMinCurr-1) * inDataStrideRow + (y+yMinCurr-1)) * cornerIsValid];

        *outData = outValue;
    }
}

extern "C"
void forwardNoNormReplicateCuda(THCState *state,
    float *intData, int intDataStrideChannel, float *outData,
    int batchSize, int nInputPlane, int nWindows, int h, int w,
    float *xMin, float *xMax, float *yMin, float *yMax,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        strided::forwardNoNormReplicateCuda(state,
            intData, intDataStrideChannel, outData,
            batchSize, nInputPlane, nWindows, h, w,
            xMin, xMax, yMin, yMax,
            strideH, strideW);
        return;
    }

    const int threadsNeeded = batchSize * nInputPlane * nWindows * h * w;
    const int numBlocks = (threadsNeeded + NUM_THREADS - 1) / NUM_THREADS;

    forwardNoNormReplicateKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
        intData, intDataStrideChannel, outData,
        batchSize, nInputPlane, nWindows, h, w,
        xMin, xMax, yMin, yMax);
    THCudaCheck(cudaGetLastError());
}

extern "C"
void forwardNoNormReplicateFracCuda(THCState *state,
    float *intData, int intDataStrideChannel, float *outData,
    int batchSize, int nInputPlane, int nWindows, int h, int w,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStrideRow, int inDataStrideChannel,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        strided::forwardNoNormReplicateFracCuda(state,
            intData, intDataStrideChannel, outData,
            batchSize, nInputPlane, nWindows, h, w,
            xMin, xMax, yMin, yMax,
            inData, inDataStrideRow, inDataStrideChannel,
            strideH, strideW);
        return;
    }

    const int threadsNeeded = batchSize * nInputPlane * nWindows * h * w;
    const int numBlocks = (threadsNeeded + NUM_THREADS - 1) / NUM_THREADS;

    forwardNoNormReplicateFracKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
        intData, intDataStrideChannel, outData,
        batchSize, nInputPlane, nWindows, h, w,
        xMin, xMax, yMin, yMax,
        inData, inDataStrideRow, inDataStrideChannel);
    THCudaCheck(cudaGetLastError());
}

/************************ updateGradInput ************************/

/************** Planewise *************/

__global__ void updateGradInputReplicatePlanewiseKernel(
    float *gradOutputIntData, float *gradInputData,
    int h, int w, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax) {

    const int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;

    if (x < h and y < w) {

        int xMinCurr, xMaxCurr, yMinCurr, yMaxCurr;
        double outValue = 0;

        for (int windowIdx = 0; windowIdx < nWindows; ++windowIdx) {

            xMinCurr = (int)ceil(-xMax[windowIdx]);
            yMinCurr = (int)ceil(-yMax[windowIdx]);

            xMaxCurr = (int)floor(-xMin[windowIdx]) + 1;
            yMaxCurr = (int)floor(-yMin[windowIdx]) + 1;

            // The following code block implements these lines
            // as if they were executed simultaneously (see `void updateGradInputFrac()`):
            // xMinCurr = (x == 0   and xMaxCurr >= 0 ? 0    : xMinCurr);
            // xMaxCurr = (x == h-1 and xMinCurr <= 0 ? h+66 : xMaxCurr);
            // yMinCurr = (y == 0   and yMaxCurr >= 0 ? 0    : yMinCurr);
            // yMaxCurr = (y == w-1 and yMinCurr <= 0 ? w+66 : yMaxCurr);

            bool needToChangeMin, needToChangeMax;

            needToChangeMin = x == 0   and xMaxCurr >= 0;
            needToChangeMax = x == h-1 and xMinCurr <= 0;
            if (needToChangeMin) xMinCurr = 0;
            if (needToChangeMax) xMaxCurr = h+66;

            needToChangeMin = y == 0   and yMaxCurr >= 0;
            needToChangeMax = y == w-1 and yMinCurr <= 0;
            if (needToChangeMin) yMinCurr = 0;
            if (needToChangeMax) yMaxCurr = w+66;

            const int t = max(0, min(x+xMinCurr, h) );
            const int b = max(0, min(x+xMaxCurr, h) );
            const int l = max(0, min(y+yMinCurr, w) );
            const int r = max(0, min(y+yMaxCurr, w) );

            outValue += gradOutputIntData[b*(w+1) + r];
            outValue -= gradOutputIntData[t*(w+1) + r];
            outValue -= gradOutputIntData[b*(w+1) + l];
            outValue += gradOutputIntData[t*(w+1) + l];

            // go to the next channel
            gradOutputIntData += (h+1)*(w+1);
        }

        gradInputData[x*w + y] = outValue;
    }
}

__global__ void updateGradInputReplicatePlanewiseFracKernel(
    float *gradOutputIntData, float *gradInputData,
    int h, int w, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *gradOutputData, int gradOutputStrideRow, int gradOutputStrideChannel) {

    const int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;

    if (x < h and y < w) {

        int xMinCurr, xMaxCurr, yMinCurr, yMaxCurr;
        double outValue = 0;

        for (int windowIdx = 0; windowIdx < nWindows; ++windowIdx) {

            xMinCurr = (int)ceil(-xMax[windowIdx]);
            yMinCurr = (int)ceil(-yMax[windowIdx]);
            const float xMinCurrFrac = (float)xMinCurr + xMax[windowIdx];
            const float yMinCurrFrac = (float)yMinCurr + yMax[windowIdx];

            xMaxCurr = (int)floor(-xMin[windowIdx]) + 1;
            yMaxCurr = (int)floor(-yMin[windowIdx]) + 1;
            const float xMaxCurrFrac = -xMin[windowIdx] + 1 - xMaxCurr;
            const float yMaxCurrFrac = -yMin[windowIdx] + 1 - yMaxCurr;

            // The following code block implements these lines
            // as if they were executed simultaneously (see `void updateGradInputFrac()`):
            // xMinCurr = (x == 0   and xMaxCurr >= 0 ? 0    : xMinCurr);
            // xMaxCurr = (x == h-1 and xMinCurr <= 0 ? h+66 : xMaxCurr);
            // yMinCurr = (y == 0   and yMaxCurr >= 0 ? 0    : yMinCurr);
            // yMaxCurr = (y == w-1 and yMinCurr <= 0 ? w+66 : yMaxCurr);

            bool needToChangeMin, needToChangeMax;

            needToChangeMin = x == 0   and xMaxCurr >= 0;
            needToChangeMax = x == h-1 and xMinCurr <= 0;
            if (needToChangeMin) xMinCurr = 0;
            if (needToChangeMax) xMaxCurr = h+66;

            needToChangeMin = y == 0   and yMaxCurr >= 0;
            needToChangeMax = y == w-1 and yMinCurr <= 0;
            if (needToChangeMin) yMinCurr = 0;
            if (needToChangeMax) yMaxCurr = w+66;

            const int t = max(0, min(x+xMinCurr, h) );
            const int b = max(0, min(x+xMaxCurr, h) );
            const int l = max(0, min(y+yMinCurr, w) );
            const int r = max(0, min(y+yMaxCurr, w) );

            const int tAdv = x+xMinCurr-1 <  h ? max(0, min(t-1, h)) : t;
            const int bAdv = x+xMaxCurr   >= 0 ? max(0, min(b+1, h)) : b;
            const int lAdv = y+yMinCurr-1 <  w ? max(0, min(l-1, w)) : l;
            const int rAdv = y+yMaxCurr   >= 0 ? max(0, min(r+1, w)) : r;

            // TODO: 1D grid
            outValue += gradOutputIntData[b*(w+1) + r];
            outValue -= gradOutputIntData[t*(w+1) + r];
            outValue -= gradOutputIntData[b*(w+1) + l];
            outValue += gradOutputIntData[t*(w+1) + l];

            // -- xMax border
            outValue +=
                ( gradOutputIntData[bAdv*(w+1) + r]
                - gradOutputIntData[b   *(w+1) + r]
                - gradOutputIntData[bAdv*(w+1) + l]
                + gradOutputIntData[b   *(w+1) + l]
                ) * xMaxCurrFrac;

            // -- yMax border
            outValue +=
                ( gradOutputIntData[b*(w+1) + rAdv]
                - gradOutputIntData[b*(w+1) + r   ]
                - gradOutputIntData[t*(w+1) + rAdv]
                + gradOutputIntData[t*(w+1) + r   ]
                ) * yMaxCurrFrac;

            // -- xMin border
            outValue +=
                ( gradOutputIntData[t   *(w+1) + r]
                - gradOutputIntData[tAdv*(w+1) + r]
                - gradOutputIntData[t   *(w+1) + l]
                + gradOutputIntData[tAdv*(w+1) + l]
                ) * xMinCurrFrac;

            // -- yMin border
            outValue +=
                ( gradOutputIntData[b*(w+1) + l   ]
                - gradOutputIntData[b*(w+1) + lAdv]
                - gradOutputIntData[t*(w+1) + l   ]
                + gradOutputIntData[t*(w+1) + lAdv]
                ) * yMinCurrFrac;

            // -- corner pixels
            outValue += 
                xMaxCurrFrac*yMaxCurrFrac * (
                   (x+xMaxCurr > h-1 or
                    y+yMaxCurr > w-1 or
                    x+xMaxCurr < 0   or
                    y+yMaxCurr < 0   or
                    b == bAdv or
                    r == rAdv) ? 0 : 
                    gradOutputData[b*gradOutputStrideRow + r]);

            outValue +=
                xMinCurrFrac*yMaxCurrFrac * (
                   (x+xMinCurr-1 > h-1 or
                    y+yMaxCurr   > w-1 or
                    x+xMinCurr-1 < 0   or
                    y+yMaxCurr   < 0   or
                    t == tAdv or
                    r == rAdv) ? 0 : 
                    gradOutputData[tAdv*gradOutputStrideRow + r]);

            outValue +=
                xMaxCurrFrac*yMinCurrFrac * (
                   (x+xMaxCurr   > h-1 or
                    y+yMinCurr-1 > w-1 or
                    x+xMaxCurr   < 0   or
                    y+yMinCurr-1 < 0   or
                    b == bAdv or
                    l == lAdv) ? 0 : 
                    gradOutputData[b*gradOutputStrideRow + lAdv]);

            outValue +=
                xMinCurrFrac*yMinCurrFrac * (
                   (x+xMinCurr-1 > h-1 or
                    y+yMinCurr-1 > w-1 or
                    x+xMinCurr-1 < 0   or
                    y+yMinCurr-1 < 0   or
                    t == tAdv or
                    l == lAdv) ? 0 : 
                    gradOutputData[tAdv*gradOutputStrideRow + lAdv]);

            // go to the next channel
            gradOutputIntData += (h+1)*(w+1);
            gradOutputData += gradOutputStrideChannel;
        }

        gradInputData[x*w + y] = outValue;
    }
}

extern "C"
void updateGradInputReplicatePlanewiseCuda(THCState *state,
    float *gradOutputIntData, float *gradInputData,
    int h, int w, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        strided::updateGradInputReplicatePlanewiseCuda(
            gradOutputIntData, gradInputData, h, w, nWindows,
            xMin, xMax, yMin, yMax, strideH, strideW);
        return;
    }

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_CHANNELS);
    dim3 dimGrid(
        (h + dimBlock.x - 1) / dimBlock.x, 
        (w + dimBlock.y - 1) / dimBlock.y);

    updateGradInputReplicatePlanewiseKernel <<<dimGrid, dimBlock, 0, THCState_getCurrentStream(state)>>> (
        gradOutputIntData, gradInputData,
        h, w, nWindows,
        xMin, xMax, yMin, yMax);
    THCudaCheck(cudaGetLastError());
}

extern "C"
void updateGradInputReplicatePlanewiseFracCuda(THCState *state,
    float *gradOutputIntData, float *gradInputData,
    int h, int w, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *gradOutputData, int gradOutputStrideRow, int gradOutputStrideChannel,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        strided::updateGradInputReplicatePlanewiseFracCuda(
            gradOutputIntData, gradInputData, h, w, nWindows,
            xMin, xMax, yMin, yMax,
            gradOutputData, gradOutputStrideRow, gradOutputStrideChannel,
            strideH, strideW);
        return;
    }

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_CHANNELS);
    dim3 dimGrid(
        (h + dimBlock.x - 1) / dimBlock.x, 
        (w + dimBlock.y - 1) / dimBlock.y);

    updateGradInputReplicatePlanewiseFracKernel <<<dimGrid, dimBlock, 0, THCState_getCurrentStream(state)>>> (
        gradOutputIntData, gradInputData,
        h, w, nWindows,
        xMin, xMax, yMin, yMax,
        gradOutputData, gradOutputStrideRow, gradOutputStrideChannel);
    THCudaCheck(cudaGetLastError());
}

/****************** Single-kernel updateGradInput (faster) **************/

__global__ void updateGradInputReplicateKernel(
    const float *gradOutputIntData, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows,
    const int h, const int w,
    const float *const xMin, const float *const xMax,
    const float *const yMin, const float *const yMax) {

    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    tmpArray += id; // tmpArray now points to our output pixel

    const int y = id % w; id /= w;
    const int x = id % h; id /= h;
    const int globalWindowIdx = id % (nInputPlane * nWindows);

    // `id` is now the current plane number
    gradOutputIntData += id * (w+1) * (h+1);

    if (id < batchSize * nInputPlane * nWindows) {

        float outValue = 0;
        int xMinCurr, xMaxCurr, yMinCurr, yMaxCurr;

        xMinCurr = (int)ceil(-xMax[globalWindowIdx]);
        yMinCurr = (int)ceil(-yMax[globalWindowIdx]);

        xMaxCurr = (int)floor(-xMin[globalWindowIdx]) + 1;
        yMaxCurr = (int)floor(-yMin[globalWindowIdx]) + 1;

        // The following code block implements these lines
        // as if they were executed simultaneously (see `void updateGradInputFrac()`):
        // xMinCurr = (x == 0   and xMaxCurr >= 0 ? 0    : xMinCurr);
        // xMaxCurr = (x == h-1 and xMinCurr <= 0 ? h+66 : xMaxCurr);
        // yMinCurr = (y == 0   and yMaxCurr >= 0 ? 0    : yMinCurr);
        // yMaxCurr = (y == w-1 and yMinCurr <= 0 ? w+66 : yMaxCurr);

        bool needToChangeMin, needToChangeMax;

        needToChangeMin = x == 0   and xMaxCurr >= 0;
        needToChangeMax = x == h-1 and xMinCurr <= 0;
        if (needToChangeMin) xMinCurr = 0;
        if (needToChangeMax) xMaxCurr = h+66;

        needToChangeMin = y == 0   and yMaxCurr >= 0;
        needToChangeMax = y == w-1 and yMinCurr <= 0;
        if (needToChangeMin) yMinCurr = 0;
        if (needToChangeMax) yMaxCurr = w+66;

        const int t = max(0, min(x+xMinCurr, h) );
        const int b = max(0, min(x+xMaxCurr, h) );
        const int l = max(0, min(y+yMinCurr, w) );
        const int r = max(0, min(y+yMaxCurr, w) );

        outValue += gradOutputIntData[b*(w+1) + r];
        outValue -= gradOutputIntData[t*(w+1) + r];
        outValue -= gradOutputIntData[b*(w+1) + l];
        outValue += gradOutputIntData[t*(w+1) + l];

        *tmpArray = outValue;
    }
}

__global__ void updateGradInputReplicateFracKernel(
    const float *gradOutputIntData, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows,
    const int h, const int w,
    const float *const xMin, const float *const xMax,
    const float *const yMin, const float *const yMax,
    const float *gradOutputData,
    const int gradOutputStrideRow, const int gradOutputStrideChannel) {

    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    tmpArray += id; // tmpArray now points to our output pixel

    const int y = id % w; id /= w;
    const int x = id % h; id /= h;
    const int globalWindowIdx = id % (nInputPlane * nWindows);

    // `id` is now the current plane number
    gradOutputIntData += id * (w+1) * (h+1);
    gradOutputData += id * gradOutputStrideChannel;

    if (id < batchSize * nInputPlane * nWindows) {

        float outValue = 0;
        int xMinCurr, xMaxCurr, yMinCurr, yMaxCurr;

        xMinCurr = (int)ceil(-xMax[globalWindowIdx]);
        yMinCurr = (int)ceil(-yMax[globalWindowIdx]);
        const float xMinCurrFrac = (float)xMinCurr + xMax[globalWindowIdx];
        const float yMinCurrFrac = (float)yMinCurr + yMax[globalWindowIdx];

        xMaxCurr = (int)floor(-xMin[globalWindowIdx]) + 1;
        yMaxCurr = (int)floor(-yMin[globalWindowIdx]) + 1;
        const float xMaxCurrFrac = -xMin[globalWindowIdx] + 1 - xMaxCurr;
        const float yMaxCurrFrac = -yMin[globalWindowIdx] + 1 - yMaxCurr;

        // The following code block implements these lines
        // as if they were executed simultaneously (see `void updateGradInputFrac()`):
        // xMinCurr = (x == 0   and xMaxCurr >= 0 ? 0    : xMinCurr);
        // xMaxCurr = (x == h-1 and xMinCurr <= 0 ? h+66 : xMaxCurr);
        // yMinCurr = (y == 0   and yMaxCurr >= 0 ? 0    : yMinCurr);
        // yMaxCurr = (y == w-1 and yMinCurr <= 0 ? w+66 : yMaxCurr);

        bool needToChangeMin, needToChangeMax;

        needToChangeMin = x == 0   and xMaxCurr >= 0;
        needToChangeMax = x == h-1 and xMinCurr <= 0;
        if (needToChangeMin) xMinCurr = 0;
        if (needToChangeMax) xMaxCurr = h+66;

        needToChangeMin = y == 0   and yMaxCurr >= 0;
        needToChangeMax = y == w-1 and yMinCurr <= 0;
        if (needToChangeMin) yMinCurr = 0;
        if (needToChangeMax) yMaxCurr = w+66;

        const int t = max(0, min(x+xMinCurr, h) );
        const int b = max(0, min(x+xMaxCurr, h) );
        const int l = max(0, min(y+yMinCurr, w) );
        const int r = max(0, min(y+yMaxCurr, w) );

        const int tAdv = x+xMinCurr-1 <  h ? max(0, min(t-1, h)) : t;
        const int bAdv = x+xMaxCurr   >= 0 ? max(0, min(b+1, h)) : b;
        const int lAdv = y+yMinCurr-1 <  w ? max(0, min(l-1, w)) : l;
        const int rAdv = y+yMaxCurr   >= 0 ? max(0, min(r+1, w)) : r;

        outValue += gradOutputIntData[b*(w+1) + r];
        outValue -= gradOutputIntData[t*(w+1) + r];
        outValue -= gradOutputIntData[b*(w+1) + l];
        outValue += gradOutputIntData[t*(w+1) + l];

        // -- xMax border
        outValue +=
            ( gradOutputIntData[bAdv*(w+1) + r]
            - gradOutputIntData[b   *(w+1) + r]
            - gradOutputIntData[bAdv*(w+1) + l]
            + gradOutputIntData[b   *(w+1) + l]
            ) * xMaxCurrFrac;

        // -- yMax border
        outValue +=
            ( gradOutputIntData[b*(w+1) + rAdv]
            - gradOutputIntData[b*(w+1) + r   ]
            - gradOutputIntData[t*(w+1) + rAdv]
            + gradOutputIntData[t*(w+1) + r   ]
            ) * yMaxCurrFrac;

        // -- xMin border
        outValue +=
            ( gradOutputIntData[t   *(w+1) + r]
            - gradOutputIntData[tAdv*(w+1) + r]
            - gradOutputIntData[t   *(w+1) + l]
            + gradOutputIntData[tAdv*(w+1) + l]
            ) * xMinCurrFrac;

        // -- yMin border
        outValue +=
            ( gradOutputIntData[b*(w+1) + l   ]
            - gradOutputIntData[b*(w+1) + lAdv]
            - gradOutputIntData[t*(w+1) + l   ]
            + gradOutputIntData[t*(w+1) + lAdv]
            ) * yMinCurrFrac;

        // -- corner pixels
        outValue += 
            xMaxCurrFrac*yMaxCurrFrac * (
               (x+xMaxCurr > h-1 or
                y+yMaxCurr > w-1 or
                x+xMaxCurr < 0   or
                y+yMaxCurr < 0   or
                b == bAdv or
                r == rAdv) ? 0 : 
                gradOutputData[b*gradOutputStrideRow + r]);

        outValue +=
            xMinCurrFrac*yMaxCurrFrac * (
               (x+xMinCurr-1 > h-1 or
                y+yMaxCurr   > w-1 or
                x+xMinCurr-1 < 0   or
                y+yMaxCurr   < 0   or
                t == tAdv or
                r == rAdv) ? 0 : 
                gradOutputData[tAdv*gradOutputStrideRow + r]);

        outValue +=
            xMaxCurrFrac*yMinCurrFrac * (
               (x+xMaxCurr   > h-1 or
                y+yMinCurr-1 > w-1 or
                x+xMaxCurr   < 0   or
                y+yMinCurr-1 < 0   or
                b == bAdv or
                l == lAdv) ? 0 : 
                gradOutputData[b*gradOutputStrideRow + lAdv]);

        outValue +=
            xMinCurrFrac*yMinCurrFrac * (
               (x+xMinCurr-1 > h-1 or
                y+yMinCurr-1 > w-1 or
                x+xMinCurr-1 < 0   or
                y+yMinCurr-1 < 0   or
                t == tAdv or
                l == lAdv) ? 0 : 
                gradOutputData[tAdv*gradOutputStrideRow + lAdv]);

        *tmpArray = outValue;
    }
}

extern "C"
void updateGradInputReplicateCuda(THCState *state,
    const float *gradOutputIntData, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *const xMin, const float *const xMax,
    const float *const yMin, const float *const yMax,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        THError("NYI");
        // strided::updateGradInputPlanewiseFracCuda(
        //     gradOutputIntData, gradInputData, h, w, nWindows,
        //     xMin, xMax, yMin, yMax,
        //     gradOutputData, gradOutputStrideRow, gradOutputStrideChannel,
        //     strideH, strideW);
        return;
    }

    const int threadsNeeded = batchSize * nInputPlane * nWindows * h * w;
    const int numBlocks = (threadsNeeded + NUM_THREADS - 1) / NUM_THREADS;

    updateGradInputReplicateKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
        gradOutputIntData, tmpArray,
        batchSize, nInputPlane, nWindows,
        h, w, xMin, xMax, yMin, yMax);
    THCudaCheck(cudaGetLastError());
}

extern "C"
void updateGradInputReplicateFracCuda(THCState *state,
    const float *gradOutputIntData, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *const xMin, const float *const xMax,
    const float *const yMin, const float *const yMax,
    const float *gradOutputData,
    const int gradOutputStrideRow, const int gradOutputStrideChannel,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        THError("NYI");
        // strided::updateGradInputPlanewiseFracCuda(
        //     gradOutputIntData, gradInputData, h, w, nWindows,
        //     xMin, xMax, yMin, yMax,
        //     gradOutputData, gradOutputStrideRow, gradOutputStrideChannel,
        //     strideH, strideW);
        return;
    }

    const int threadsNeeded = batchSize * nInputPlane * nWindows * h * w;
    const int numBlocks = (threadsNeeded + NUM_THREADS - 1) / NUM_THREADS;

    updateGradInputReplicateFracKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
        gradOutputIntData, tmpArray,
        batchSize, nInputPlane, nWindows,
        h, w, xMin, xMax, yMin, yMax,
        gradOutputData, gradOutputStrideRow, gradOutputStrideChannel);
    THCudaCheck(cudaGetLastError());
}

/************************ accGradParameters planewise ************************/

__global__ void xMaxDeltaIntegralReplicatePlanewiseFracKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMax, const float *yMin, const float *yMax,
    const float *inData, const int inDataStrideRow) {
 
    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and x <= h and y <= w) {

        tmpArray += windowIdx * h * w;

        // const int xMinInt = (int)ceil(xMin[windowIdx]-1);
        // const float xMinFrac = xMinInt-xMin[windowIdx]+1;

        const int yMinInt = (int)ceil(yMin[windowIdx]-1);
        const float yMinFrac = yMinInt-yMin[windowIdx]+1;

        const int xMaxInt = (int)floor(xMax[windowIdx]);
        // const float xMaxFrac = xMax[windowIdx]-xMaxInt;

        const int yMaxInt = (int)floor(yMax[windowIdx]);
        const float yMaxFrac = yMax[windowIdx]-yMaxInt;

        // const float tlCorner = y+yMinInt <  1 or x+xMinInt <  1 ? 0 :
        //                      inData[
        //                         max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
        //                         max(0,min(w-1,y+yMinInt-1))];
        const float blCorner = y+yMinInt <  1 or x+xMaxInt >= h ? 0 :
                            inData[
                                max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
                                max(0,min(w-1,y+yMinInt-1))];
        // const float trCorner = y+yMaxInt >= w or x+xMinInt <  1 ? 0 :
        //                      inData[
        //                         max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
        //                         max(0,min(w-1,y+yMaxInt  ))];
        const float brCorner = y+yMaxInt >= w or x+xMaxInt >= h ? 0 :
                            inData[
                                max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
                                max(0,min(w-1,y+yMaxInt  ))];

        float delta = 0;

        delta += brCorner * (y+yMaxInt <  1 ? 1.0f : yMaxFrac);
        delta += blCorner * (y+yMinInt >= w ? 1.0f : yMinFrac);

        delta += 
            intData[max(0,min(x+xMaxInt+1, h))*(w+1) 
                  + max(0,min(y+yMaxInt, w))];
        delta -=
            intData[max(0,min(x+xMaxInt  , h))*(w+1) 
                  + max(0,min(y+yMaxInt, w))];
        delta -=
            intData[max(0,min(x+xMaxInt+1, h))*(w+1)
                  + max(0,min(y+yMinInt, w))];
        delta +=
            intData[max(0,min(x+xMaxInt  , h))*(w+1)
                  + max(0,min(y+yMinInt, w))];

        delta *= (x+xMaxInt >= 1 and x+xMaxInt < h);
        tmpArray[(x-1)*w + (y-1)] *= delta;
    }
}

__global__ void xMinDeltaIntegralReplicatePlanewiseFracKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *yMin, const float *yMax,
    const float *inData, const int inDataStrideRow) {
 
    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and x <= h and y <= w) {

        tmpArray += windowIdx * h * w;

        const int xMinInt = (int)ceil(xMin[windowIdx]-1);
        // const float xMinFrac = xMinInt-xMin[windowIdx]+1;

        const int yMinInt = (int)ceil(yMin[windowIdx]-1);
        const float yMinFrac = yMinInt-yMin[windowIdx]+1;

        // const int xMaxInt = (int)floor(xMax[windowIdx]);
        // const float xMaxFrac = xMax[windowIdx]-xMaxInt;

        const int yMaxInt = (int)floor(yMax[windowIdx]);
        const float yMaxFrac = yMax[windowIdx]-yMaxInt;

        const float tlCorner = y+yMinInt <  1 or x+xMinInt <  1 ? 0 :
                             inData[
                                max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
                                max(0,min(w-1,y+yMinInt-1))];
        // const float blCorner = y+yMinInt <  1 or x+xMaxInt >= h ? 0 :
        //                     inData[
        //                         max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
        //                         max(0,min(w-1,y+yMinInt-1))];
        const float trCorner = y+yMaxInt >= w or x+xMinInt <  1 ? 0 :
                             inData[
                                max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
                                max(0,min(w-1,y+yMaxInt  ))];
        // const float brCorner = y+yMaxInt >= w or x+xMaxInt >= h ? 0 :
        //                     inData[
        //                         max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
        //                         max(0,min(w-1,y+yMaxInt  ))];

        float delta = 0;

        delta += trCorner * (y+yMaxInt <  1 ? 1.0f : yMaxFrac);
        delta += tlCorner * (y+yMinInt >= w ? 1.0f : yMinFrac);

        delta += 
            intData[max(0,min(x+xMinInt  , h))*(w+1)
                  + max(0,min(y+yMaxInt, w))];
        delta -=
            intData[max(0,min(x+xMinInt-1, h))*(w+1)
                  + max(0,min(y+yMaxInt, w))];
        delta -=
            intData[max(0,min(x+xMinInt  , h))*(w+1)
                  + max(0,min(y+yMinInt, w))];
        delta +=
            intData[max(0,min(x+xMinInt-1, h))*(w+1)
                  + max(0,min(y+yMinInt, w))];

        delta *= (x+xMinInt >= 1 and x+xMinInt < h);
        tmpArray[(x-1)*w + (y-1)] *= -delta;
    }
}

__global__ void yMaxDeltaIntegralReplicatePlanewiseFracKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMax,
    const float *inData, const int inDataStrideRow) {
 
    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and x <= h and y <= w) {

        tmpArray += windowIdx * h * w;

        const int xMinInt = (int)ceil(xMin[windowIdx]-1);
        const float xMinFrac = xMinInt-xMin[windowIdx]+1;

        // const int yMinInt = (int)ceil(yMin[windowIdx]-1);
        // const float yMinFrac = yMinInt-yMin[windowIdx]+1;

        const int xMaxInt = (int)floor(xMax[windowIdx]);
        const float xMaxFrac = xMax[windowIdx]-xMaxInt;

        const int yMaxInt = (int)floor(yMax[windowIdx]);
        // const float yMaxFrac = yMax[windowIdx]-yMaxInt;

        // const float tlCorner = y+yMinInt <  1 or x+xMinInt <  1 ? 0 :
        //                      inData[
        //                         max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
        //                         max(0,min(w-1,y+yMinInt-1))];
        // const float blCorner = y+yMinInt <  1 or x+xMaxInt >= h ? 0 :
        //                     inData[
        //                         max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
        //                         max(0,min(w-1,y+yMinInt-1))];
        const float trCorner = y+yMaxInt >= w or x+xMinInt <  1 ? 0 :
                             inData[
                                max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
                                max(0,min(w-1,y+yMaxInt  ))];
        const float brCorner = y+yMaxInt >= w or x+xMaxInt >= h ? 0 :
                            inData[
                                max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
                                max(0,min(w-1,y+yMaxInt  ))];

        float delta = 0;

        delta += trCorner * (x+xMinInt >= h ? 1.0f : xMinFrac);
        delta += brCorner * (x+xMaxInt <  1 ? 1.0f : xMaxFrac);

        delta += 
            intData[max(0,min(x+xMaxInt, h))*(w+1)
                  + max(0,min(y+yMaxInt+1, w))];
        delta -=
            intData[max(0,min(x+xMaxInt, h))*(w+1)
                  + max(0,min(y+yMaxInt  , w))];
        delta -=
            intData[max(0,min(x+xMinInt, h))*(w+1)
                  + max(0,min(y+yMaxInt+1, w))];
        delta +=
            intData[max(0,min(x+xMinInt, h))*(w+1)
                  + max(0,min(y+yMaxInt  , w))];

        delta *= (y+yMaxInt >= 1 and y+yMaxInt < w);
        tmpArray[(x-1)*w + (y-1)] *= delta;
    }
}

__global__ void yMinDeltaIntegralReplicatePlanewiseFracKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMin,
    const float *inData, const int inDataStrideRow) {
 
    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and x <= h and y <= w) {

        tmpArray += windowIdx * h * w;

        const int xMinInt = (int)ceil(xMin[windowIdx]-1);
        const float xMinFrac = xMinInt-xMin[windowIdx]+1;

        const int yMinInt = (int)ceil(yMin[windowIdx]-1);
        // const float yMinFrac = yMinInt-yMin[windowIdx]+1;

        const int xMaxInt = (int)floor(xMax[windowIdx]);
        const float xMaxFrac = xMax[windowIdx]-xMaxInt;

        // const int yMaxInt = (int)floor(yMax[windowIdx]);
        // const float yMaxFrac = yMax[windowIdx]-yMaxInt;

        const float tlCorner = y+yMinInt <  1 or x+xMinInt <  1 ? 0 :
                             inData[
                                max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
                                max(0,min(w-1,y+yMinInt-1))];
        const float blCorner = y+yMinInt <  1 or x+xMaxInt >= h ? 0 :
                            inData[
                                max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
                                max(0,min(w-1,y+yMinInt-1))];
        // const float trCorner = y+yMaxInt >= w or x+xMinInt <  1 ? 0 :
        //                      inData[
        //                         max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
        //                         max(0,min(w-1,y+yMaxInt  ))];
        // const float brCorner = y+yMaxInt >= w or x+xMaxInt >= h ? 0 :
        //                     inData[
        //                         max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
        //                         max(0,min(w-1,y+yMaxInt  ))];

        float delta = 0;

        delta += tlCorner * (x+xMinInt >= h ? 1.0f : xMinFrac);
        delta += blCorner * (x+xMaxInt <  1 ? 1.0f : xMaxFrac);

        delta += 
            intData[max(0,min(x+xMaxInt, h))*(w+1)
                  + max(0,min(y+yMinInt  , w))];
        delta -=
            intData[max(0,min(x+xMaxInt, h))*(w+1)
                  + max(0,min(y+yMinInt-1, w))];
        delta -=
            intData[max(0,min(x+xMinInt, h))*(w+1)
                  + max(0,min(y+yMinInt  , w))];
        delta +=
            intData[max(0,min(x+xMinInt, h))*(w+1)
                  + max(0,min(y+yMinInt-1, w))];

        delta *= (y+yMinInt >= 1 and y+yMinInt < w);
        tmpArray[(x-1)*w + (y-1)] *= -delta;
    }
}

extern "C"
void backwardReplicatePlanewiseFracCuda(THCState *state,
    float *intData, float *tmpArray,
    int nWindows, int h, int w,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStrideRow,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        strided::backwardReplicateFracCuda(
            intData, tmpArray, nWindows, h, w,
            xMin, xMax, yMin, yMax, inData, inDataStrideRow,
            strideH, strideW);
        return;
    }

    dim3 dimBlock(NUM_THREADS);
    dim3 dimGrid((nWindows * h * w + dimBlock.x - 1) / dimBlock.x);

    xMaxDeltaIntegralReplicatePlanewiseFracKernel <<<dimGrid, dimBlock, 0, THCState_getCurrentStream(state)>>> (
        intData, tmpArray + 0*nWindows*h*w, nWindows, h, w,
        xMax, yMin, yMax, inData, inDataStrideRow);

    xMinDeltaIntegralReplicatePlanewiseFracKernel <<<dimGrid, dimBlock, 0, THCState_getCurrentStream(state)>>> (
        intData, tmpArray + 1*nWindows*h*w, nWindows, h, w,
        xMin, yMin, yMax, inData, inDataStrideRow);

    yMaxDeltaIntegralReplicatePlanewiseFracKernel <<<dimGrid, dimBlock, 0, THCState_getCurrentStream(state)>>> (
        intData, tmpArray + 2*nWindows*h*w, nWindows, h, w,
        xMin, xMax, yMax, inData, inDataStrideRow);

    yMinDeltaIntegralReplicatePlanewiseFracKernel <<<dimGrid, dimBlock, 0, THCState_getCurrentStream(state)>>> (
        intData, tmpArray + 3*nWindows*h*w, nWindows, h, w,
        xMin, xMax, yMin, inData, inDataStrideRow);
    THCudaCheck(cudaGetLastError());
}

__global__ void xMaxDeltaIntegralReplicatePlanewiseKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMax, const float *yMin, const float *yMax) {
 
    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and x <= h and y <= w) {

        tmpArray += windowIdx * h * w;

        // const int xMinInt = (int)ceil(xMin[windowIdx]-1);
        const int yMinInt = (int)ceil(yMin[windowIdx]-1);
        const int xMaxInt = (int)floor(xMax[windowIdx]);
        const int yMaxInt = (int)floor(yMax[windowIdx]);

        float delta = 0;

        delta += 
            intData[max(1,min(x+xMaxInt+1, h))*(w+1) 
                  + max(0,min(y+yMaxInt, w))];
        delta -=
            intData[max(0,min(x+xMaxInt  , h))*(w+1) 
                  + max(0,min(y+yMaxInt, w))];
        delta -=
            intData[max(1,min(x+xMaxInt+1, h))*(w+1)
                  + max(0,min(y+yMinInt, w))];
        delta +=
            intData[max(0,min(x+xMaxInt  , h))*(w+1)
                  + max(0,min(y+yMinInt, w))];

        delta *= (x+xMaxInt >= 1 and x+xMaxInt < h);
        tmpArray[(x-1)*w + (y-1)] *= delta;
    }
}

__global__ void xMinDeltaIntegralReplicatePlanewiseKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *yMin, const float *yMax) {
 
    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and x <= h and y <= w) {

        tmpArray += windowIdx * h * w;

        const int xMinInt = (int)ceil(xMin[windowIdx]-1);
        const int yMinInt = (int)ceil(yMin[windowIdx]-1);
        // const int xMaxInt = (int)floor(xMax[windowIdx]);
        const int yMaxInt = (int)floor(yMax[windowIdx]);

        float delta = 0;

        delta += 
            intData[max(0,min(x+xMinInt  , h-1))*(w+1)
                  + max(0,min(y+yMaxInt, w))];
        delta -=
            intData[max(0,min(x+xMinInt-1, h  ))*(w+1)
                  + max(0,min(y+yMaxInt, w))];
        delta -=
            intData[max(0,min(x+xMinInt  , h-1))*(w+1)
                  + max(0,min(y+yMinInt, w))];
        delta +=
            intData[max(0,min(x+xMinInt-1, h  ))*(w+1)
                  + max(0,min(y+yMinInt, w))];

        delta *= (x+xMinInt >= 1 and x+xMinInt < h);
        tmpArray[(x-1)*w + (y-1)] *= -delta;
    }
}

__global__ void yMaxDeltaIntegralReplicatePlanewiseKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMax) {
 
    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and x <= h and y <= w) {

        tmpArray += windowIdx * h * w;

        const int xMinInt = (int)ceil(xMin[windowIdx]-1);
        // const int yMinInt = (int)ceil(yMin[windowIdx]-1);
        const int xMaxInt = (int)floor(xMax[windowIdx]);
        const int yMaxInt = (int)floor(yMax[windowIdx]);

        float delta = 0;

        delta += 
            intData[max(0,min(x+xMaxInt, h))*(w+1)
                  + max(1,min(y+yMaxInt+1, w))];
        delta -=
            intData[max(0,min(x+xMaxInt, h))*(w+1)
                  + max(0,min(y+yMaxInt  , w))];
        delta -=
            intData[max(0,min(x+xMinInt, h))*(w+1)
                  + max(1,min(y+yMaxInt+1, w))];
        delta +=
            intData[max(0,min(x+xMinInt, h))*(w+1)
                  + max(0,min(y+yMaxInt  , w))];

        delta *= (y+yMaxInt >= 1 and y+yMaxInt < w);
        tmpArray[(x-1)*w + (y-1)] *= delta;
    }
}

__global__ void yMinDeltaIntegralReplicatePlanewiseKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMin) {
 
    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and x <= h and y <= w) {

        tmpArray += windowIdx * h * w;

        const int xMinInt = (int)ceil(xMin[windowIdx]-1);
        const int yMinInt = (int)ceil(yMin[windowIdx]-1);
        const int xMaxInt = (int)floor(xMax[windowIdx]);
        // const int yMaxInt = (int)floor(yMax[windowIdx]);

        float delta = 0;

        delta += 
            intData[max(0,min(x+xMaxInt, h))*(w+1)
                  + max(0,min(y+yMinInt  , w  ))];
        delta -=
            intData[max(0,min(x+xMaxInt, h))*(w+1)
                  + max(0,min(y+yMinInt-1, w-1))];
        delta -=
            intData[max(0,min(x+xMinInt, h))*(w+1)
                  + max(0,min(y+yMinInt  , w  ))];
        delta +=
            intData[max(0,min(x+xMinInt, h))*(w+1)
                  + max(0,min(y+yMinInt-1, w-1))];

        delta *= (y+yMinInt >= 1 and y+yMinInt < w);
        tmpArray[(x-1)*w + (y-1)] *= -delta;
    }
}

extern "C"
void backwardReplicatePlanewiseCuda(THCState *state,
    float *intData, float *tmpArray,
    int nWindows, int h, int w,
    float *xMin, float *xMax, float *yMin, float *yMax,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        strided::backwardReplicateCuda(
            intData, tmpArray, nWindows, h, w,
            xMin, xMax, yMin, yMax, strideH, strideW);
        return;
    }

    dim3 dimBlock(NUM_THREADS);
    dim3 dimGrid((nWindows * h * w + dimBlock.x - 1) / dimBlock.x);

    xMaxDeltaIntegralReplicatePlanewiseKernel <<<dimGrid, dimBlock, 0, THCState_getCurrentStream(state)>>> (
        intData, tmpArray + 0*nWindows*h*w,
        nWindows, h, w, xMax, yMin, yMax);

    xMinDeltaIntegralReplicatePlanewiseKernel <<<dimGrid, dimBlock, 0, THCState_getCurrentStream(state)>>> (
        intData, tmpArray + 1*nWindows*h*w,
        nWindows, h, w, xMin, yMin, yMax);

    yMaxDeltaIntegralReplicatePlanewiseKernel <<<dimGrid, dimBlock, 0, THCState_getCurrentStream(state)>>> (
        intData, tmpArray + 2*nWindows*h*w,
        nWindows, h, w, xMin, xMax, yMax);

    yMinDeltaIntegralReplicatePlanewiseKernel <<<dimGrid, dimBlock, 0, THCState_getCurrentStream(state)>>> (
        intData, tmpArray + 3*nWindows*h*w,
        nWindows, h, w, xMin, xMax, yMin);
    THCudaCheck(cudaGetLastError());
}

/************************ accGradParameters fastest *********************/

__global__ void xMaxDeltaIntegralReplicateFracKernel(
    const float *intData, const int intDataStrideChannel, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMax, const float *yMin, const float *yMax,
    const float *inData, const int inDataStrideRow, const int inDataStrideChannel) {
 
    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    tmpArray += id; // tmpArray now points to our output pixel

    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int windowIdx = id % nWindows; id /= nWindows;

    // `id` is now is now the current global input plane number
    intData  += id * intDataStrideChannel;
    inData   += id *  inDataStrideChannel;

    const int globalWindowIdx = (id % nInputPlane) * nWindows + windowIdx; id /= nInputPlane;
    const int & batchIdx = id;

    if (batchIdx < batchSize) {

        // const int xMinInt = (int)ceil(xMin[globalWindowIdx]-1);
        // const float xMinFrac = xMinInt-xMin[globalWindowIdx]+1;

        const int yMinInt = (int)ceil(yMin[globalWindowIdx]-1);
        const float yMinFrac = yMinInt-yMin[globalWindowIdx]+1;

        const int xMaxInt = (int)floor(xMax[globalWindowIdx]);
        // const float xMaxFrac = xMax[globalWindowIdx]-xMaxInt;

        const int yMaxInt = (int)floor(yMax[globalWindowIdx]);
        const float yMaxFrac = yMax[globalWindowIdx]-yMaxInt;

        // const float tlCorner = y+yMinInt <  1 or x+xMinInt <  1 ? 0 :
        //                      inData[
        //                         max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
        //                         max(0,min(w-1,y+yMinInt-1))];
        const float blCorner = y+yMinInt <  1 or x+xMaxInt >= h ? 0 :
                            inData[
                                max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
                                max(0,min(w-1,y+yMinInt-1))];
        // const float trCorner = y+yMaxInt >= w or x+xMinInt <  1 ? 0 :
        //                      inData[
        //                         max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
        //                         max(0,min(w-1,y+yMaxInt  ))];
        const float brCorner = y+yMaxInt >= w or x+xMaxInt >= h ? 0 :
                            inData[
                                max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
                                max(0,min(w-1,y+yMaxInt  ))];

        float delta = 0;

        delta += brCorner * (y+yMaxInt <  1 ? 1.0f : yMaxFrac);
        delta += blCorner * (y+yMinInt >= w ? 1.0f : yMinFrac);

        delta += 
            intData[max(0,min(x+xMaxInt+1, h))*(w+1) 
                  + max(0,min(y+yMaxInt, w))];
        delta -=
            intData[max(0,min(x+xMaxInt  , h))*(w+1) 
                  + max(0,min(y+yMaxInt, w))];
        delta -=
            intData[max(0,min(x+xMaxInt+1, h))*(w+1)
                  + max(0,min(y+yMinInt, w))];
        delta +=
            intData[max(0,min(x+xMaxInt  , h))*(w+1)
                  + max(0,min(y+yMinInt, w))];

        delta *= (x+xMaxInt >= 1 and x+xMaxInt < h);
        *tmpArray = delta;
    }
}

template <bool inputIsOnes>
__global__ void xMinDeltaIntegralReplicateFracKernel(
    const float *intData, const int intDataStrideChannel, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMin, const float *yMin, const float *yMax,
    const float *inData, const int inDataStrideRow, const int inDataStrideChannel) {
 
    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    tmpArray += id; // tmpArray now points to our output pixel

    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int windowIdx = id % nWindows; id /= nWindows;

    // `id` is now is now the current global input plane number
    intData  += id * intDataStrideChannel;
    if (not inputIsOnes) inData += id * inDataStrideChannel;

    const int globalWindowIdx = (id % nInputPlane) * nWindows + windowIdx; id /= nInputPlane;
    const int & batchIdx = id;

    if (batchIdx < batchSize) {

        const int xMinInt = (int)ceil(xMin[globalWindowIdx]-1);
        // const float xMinFrac = xMinInt-xMin[globalWindowIdx]+1;

        const int yMinInt = (int)ceil(yMin[globalWindowIdx]-1);
        const float yMinFrac = yMinInt-yMin[globalWindowIdx]+1;

        // const int xMaxInt = (int)floor(xMax[globalWindowIdx]);
        // const float xMaxFrac = xMax[globalWindowIdx]-xMaxInt;

        const int yMaxInt = (int)floor(yMax[globalWindowIdx]);
        const float yMaxFrac = yMax[globalWindowIdx]-yMaxInt;

        int valid;

        valid = not (y+yMinInt <  1) & not (x+xMinInt <  1);
        const float tlCorner = valid * (inputIsOnes ? 1 :
            inData[(max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow + max(0,min(w-1,y+yMinInt-1))) * valid]);
        // valid = not (y+yMinInt <  1) & not (x+xMaxInt >= h);
        // const float blCorner = valid * (inputIsOnes ? 1 :
        //     inData[(max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow + max(0,min(w-1,y+yMinInt-1))) * valid]);
        valid = not (y+yMaxInt >= w) & not (x+xMinInt <  1);
        const float trCorner = valid * (inputIsOnes ? 1 :
            inData[(max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow + max(0,min(w-1,y+yMaxInt  ))) * valid]);
        // valid = not (y+yMaxInt >= w) & not (x+xMaxInt >= h);
        // const float brCorner = valid * (inputIsOnes ? 1 :
        //     inData[(max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow + max(0,min(w-1,y+yMaxInt  ))) * valid]);

        float delta = 0;

        delta += trCorner * (y+yMaxInt <  1 ? 1.0f : yMaxFrac);
        delta += tlCorner * (y+yMinInt >= w ? 1.0f : yMinFrac);

        delta += 
            intData[max(0,min(x+xMinInt  , h))*(w+1)
                  + max(0,min(y+yMaxInt, w))];
        delta -=
            intData[max(0,min(x+xMinInt-1, h))*(w+1)
                  + max(0,min(y+yMaxInt, w))];
        delta -=
            intData[max(0,min(x+xMinInt  , h))*(w+1)
                  + max(0,min(y+yMinInt, w))];
        delta +=
            intData[max(0,min(x+xMinInt-1, h))*(w+1)
                  + max(0,min(y+yMinInt, w))];

        delta *= (x+xMinInt >= 1) & (x+xMinInt < h);
        *tmpArray = -delta;
    }
}

template __global__ void xMinDeltaIntegralReplicateFracKernel<false>(
    const float *intData, const int intDataStrideChannel, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMin, const float *yMin, const float *yMax,
    const float *inData, const int inDataStrideRow, const int inDataStrideChannel);
template __global__ void xMinDeltaIntegralReplicateFracKernel<true>(
    const float *intData, const int intDataStrideChannel, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMin, const float *yMin, const float *yMax,
    const float *inData, const int inDataStrideRow, const int inDataStrideChannel);

__global__ void yMaxDeltaIntegralReplicateFracKernel(
    const float *intData, const int intDataStrideChannel, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMax,
    const float *inData, const int inDataStrideRow, const int inDataStrideChannel) {
 
    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    tmpArray += id; // tmpArray now points to our output pixel

    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int windowIdx = id % nWindows; id /= nWindows;

    // `id` is now is now the current global input plane number
    intData  += id * intDataStrideChannel;
    inData   += id *  inDataStrideChannel;

    const int globalWindowIdx = (id % nInputPlane) * nWindows + windowIdx; id /= nInputPlane;
    const int & batchIdx = id;

    if (batchIdx < batchSize) {

        const int xMinInt = (int)ceil(xMin[globalWindowIdx]-1);
        const float xMinFrac = xMinInt-xMin[globalWindowIdx]+1;

        // const int yMinInt = (int)ceil(yMin[globalWindowIdx]-1);
        // const float yMinFrac = yMinInt-yMin[globalWindowIdx]+1;

        const int xMaxInt = (int)floor(xMax[globalWindowIdx]);
        const float xMaxFrac = xMax[globalWindowIdx]-xMaxInt;

        const int yMaxInt = (int)floor(yMax[globalWindowIdx]);
        // const float yMaxFrac = yMax[globalWindowIdx]-yMaxInt;

        // const float tlCorner = y+yMinInt <  1 or x+xMinInt <  1 ? 0 :
        //                      inData[
        //                         max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
        //                         max(0,min(w-1,y+yMinInt-1))];
        // const float blCorner = y+yMinInt <  1 or x+xMaxInt >= h ? 0 :
        //                     inData[
        //                         max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
        //                         max(0,min(w-1,y+yMinInt-1))];
        const float trCorner = y+yMaxInt >= w or x+xMinInt <  1 ? 0 :
                             inData[
                                max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
                                max(0,min(w-1,y+yMaxInt  ))];
        const float brCorner = y+yMaxInt >= w or x+xMaxInt >= h ? 0 :
                            inData[
                                max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
                                max(0,min(w-1,y+yMaxInt  ))];

        float delta = 0;

        delta += trCorner * (x+xMinInt >= h ? 1.0f : xMinFrac);
        delta += brCorner * (x+xMaxInt <  1 ? 1.0f : xMaxFrac);

        delta += 
            intData[max(0,min(x+xMaxInt, h))*(w+1)
                  + max(0,min(y+yMaxInt+1, w))];
        delta -=
            intData[max(0,min(x+xMaxInt, h))*(w+1)
                  + max(0,min(y+yMaxInt  , w))];
        delta -=
            intData[max(0,min(x+xMinInt, h))*(w+1)
                  + max(0,min(y+yMaxInt+1, w))];
        delta +=
            intData[max(0,min(x+xMinInt, h))*(w+1)
                  + max(0,min(y+yMaxInt  , w))];

        delta *= (y+yMaxInt >= 1 and y+yMaxInt < w);
        *tmpArray = delta;
    }
}

__global__ void yMinDeltaIntegralReplicateFracKernel(
    const float *intData, const int intDataStrideChannel, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMin,
    const float *inData, const int inDataStrideRow, const int inDataStrideChannel) {
 
    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    tmpArray += id; // tmpArray now points to our output pixel

    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int windowIdx = id % nWindows; id /= nWindows;

    // `id` is now is now the current global input plane number
    intData  += id * intDataStrideChannel;
    inData   += id *  inDataStrideChannel;

    const int globalWindowIdx = (id % nInputPlane) * nWindows + windowIdx; id /= nInputPlane;
    const int & batchIdx = id;

    if (batchIdx < batchSize) {

        const int xMinInt = (int)ceil(xMin[globalWindowIdx]-1);
        const float xMinFrac = xMinInt-xMin[globalWindowIdx]+1;

        const int yMinInt = (int)ceil(yMin[globalWindowIdx]-1);
        // const float yMinFrac = yMinInt-yMin[globalWindowIdx]+1;

        const int xMaxInt = (int)floor(xMax[globalWindowIdx]);
        const float xMaxFrac = xMax[globalWindowIdx]-xMaxInt;

        // const int yMaxInt = (int)floor(yMax[globalWindowIdx]);
        // const float yMaxFrac = yMax[globalWindowIdx]-yMaxInt;

        const float tlCorner = y+yMinInt <  1 or x+xMinInt <  1 ? 0 :
                             inData[
                                max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
                                max(0,min(w-1,y+yMinInt-1))];
        const float blCorner = y+yMinInt <  1 or x+xMaxInt >= h ? 0 :
                            inData[
                                max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
                                max(0,min(w-1,y+yMinInt-1))];
        // const float trCorner = y+yMaxInt >= w or x+xMinInt <  1 ? 0 :
        //                      inData[
        //                         max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
        //                         max(0,min(w-1,y+yMaxInt  ))];
        // const float brCorner = y+yMaxInt >= w or x+xMaxInt >= h ? 0 :
        //                     inData[
        //                         max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
        //                         max(0,min(w-1,y+yMaxInt  ))];

        float delta = 0;

        delta += tlCorner * (x+xMinInt >= h ? 1.0f : xMinFrac);
        delta += blCorner * (x+xMaxInt <  1 ? 1.0f : xMaxFrac);

        delta += 
            intData[max(0,min(x+xMaxInt, h))*(w+1)
                  + max(0,min(y+yMinInt  , w))];
        delta -=
            intData[max(0,min(x+xMaxInt, h))*(w+1)
                  + max(0,min(y+yMinInt-1, w))];
        delta -=
            intData[max(0,min(x+xMinInt, h))*(w+1)
                  + max(0,min(y+yMinInt  , w))];
        delta +=
            intData[max(0,min(x+xMinInt, h))*(w+1)
                  + max(0,min(y+yMinInt-1, w))];

        delta *= (y+yMinInt >= 1 and y+yMinInt < w);
        *tmpArray = -delta;
    }
}

extern "C"
void backwardReplicateFracCuda(THCState *state, const int paramId, const bool inputIsOnes,
    const float *intData, const int intDataStrideChannel, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMin, const float *yMax,
    const float *inData, int inDataStrideRow, const int inDataStrideChannel,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        THError("NYI");
        // strided::backwardFracCuda(
        //     intData, tmpArray, nWindows, h, w,
        //     xMin, xMax, yMin, yMax, inData, inDataStrideRow,
        //     strideH, strideW);
        return;
    }

    const int threadsNeeded = batchSize * nInputPlane * nWindows * h * w;
    const int numBlocks = (threadsNeeded + NUM_THREADS - 1) / NUM_THREADS;

    switch (paramId) {
    case 0:
        if (inputIsOnes)
        xMinDeltaIntegralReplicateFracKernel <true> <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMin, yMin, yMax,
            inData, inDataStrideRow, inDataStrideChannel);
        else
        xMinDeltaIntegralReplicateFracKernel <false> <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMin, yMin, yMax,
            inData, inDataStrideRow, inDataStrideChannel);
            break;
    case 1:
        xMaxDeltaIntegralReplicateFracKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMax, yMin, yMax,
            inData, inDataStrideRow, inDataStrideChannel); break;
    case 2:
        yMinDeltaIntegralReplicateFracKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMin, xMax, yMin,
            inData, inDataStrideRow, inDataStrideChannel); break;
    case 3:
        yMaxDeltaIntegralReplicateFracKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMin, xMax, yMax,
            inData, inDataStrideRow, inDataStrideChannel); break;
    }
    THCudaCheck(cudaGetLastError());
}

__global__ void xMaxDeltaIntegralReplicateKernel(
    const float *intData, const int intDataStrideChannel, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMax, const float *yMin, const float *yMax) {
 
    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    tmpArray += id; // tmpArray now points to our output pixel

    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int windowIdx = id % nWindows; id /= nWindows;

    // `id` is now is now the current global input plane number
    intData  += id * intDataStrideChannel;

    const int globalWindowIdx = (id % nInputPlane) * nWindows + windowIdx; id /= nInputPlane;
    const int & batchIdx = id;

    if (batchIdx < batchSize) {

        // const int xMinInt = (int)ceil(xMin[globalWindowIdx]-1);
        const int yMinInt = (int)ceil(yMin[globalWindowIdx]-1);
        const int xMaxInt = (int)floor(xMax[globalWindowIdx]);
        const int yMaxInt = (int)floor(yMax[globalWindowIdx]);

        float delta = 0;

        delta += 
            intData[max(0,min(x+xMaxInt+1, h))*(w+1) 
                  + max(0,min(y+yMaxInt, w))];
        delta -=
            intData[max(0,min(x+xMaxInt  , h))*(w+1) 
                  + max(0,min(y+yMaxInt, w))];
        delta -=
            intData[max(0,min(x+xMaxInt+1, h))*(w+1)
                  + max(0,min(y+yMinInt, w))];
        delta +=
            intData[max(0,min(x+xMaxInt  , h))*(w+1)
                  + max(0,min(y+yMinInt, w))];

        delta *= (x+xMaxInt >= 1 and x+xMaxInt < h);
        *tmpArray = delta;
    }
}

__global__ void xMinDeltaIntegralReplicateKernel(
    const float *intData, const int intDataStrideChannel, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMin, const float *yMin, const float *yMax) {
 
    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    tmpArray += id; // tmpArray now points to our output pixel

    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int windowIdx = id % nWindows; id /= nWindows;

    // `id` is now is now the current global input plane number
    intData  += id * intDataStrideChannel;

    const int globalWindowIdx = (id % nInputPlane) * nWindows + windowIdx; id /= nInputPlane;
    const int & batchIdx = id;

    if (batchIdx < batchSize) {

        const int xMinInt = (int)ceil(xMin[globalWindowIdx]-1);
        const int yMinInt = (int)ceil(yMin[globalWindowIdx]-1);
        // const int xMaxInt = (int)floor(xMax[globalWindowIdx]);
        const int yMaxInt = (int)floor(yMax[globalWindowIdx]);

        float delta = 0;

        delta += 
            intData[max(0,min(x+xMinInt  , h))*(w+1)
                  + max(0,min(y+yMaxInt, w))];
        delta -=
            intData[max(0,min(x+xMinInt-1, h))*(w+1)
                  + max(0,min(y+yMaxInt, w))];
        delta -=
            intData[max(0,min(x+xMinInt  , h))*(w+1)
                  + max(0,min(y+yMinInt, w))];
        delta +=
            intData[max(0,min(x+xMinInt-1, h))*(w+1)
                  + max(0,min(y+yMinInt, w))];

        delta *= (x+xMinInt >= 1 and x+xMinInt < h);
        *tmpArray = -delta;
    }
}

__global__ void yMaxDeltaIntegralReplicateKernel(
    const float *intData, const int intDataStrideChannel, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMax) {
 
    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    tmpArray += id; // tmpArray now points to our output pixel

    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int windowIdx = id % nWindows; id /= nWindows;

    // `id` is now is now the current global input plane number
    intData  += id * intDataStrideChannel;

    const int globalWindowIdx = (id % nInputPlane) * nWindows + windowIdx; id /= nInputPlane;
    const int & batchIdx = id;

    if (batchIdx < batchSize) {

        const int xMinInt = (int)ceil(xMin[globalWindowIdx]-1);
        // const int yMinInt = (int)ceil(yMin[globalWindowIdx]-1);
        const int xMaxInt = (int)floor(xMax[globalWindowIdx]);
        const int yMaxInt = (int)floor(yMax[globalWindowIdx]);

        float delta = 0;

        delta += 
            intData[max(0,min(x+xMaxInt, h))*(w+1)
                  + max(0,min(y+yMaxInt+1, w))];
        delta -=
            intData[max(0,min(x+xMaxInt, h))*(w+1)
                  + max(0,min(y+yMaxInt  , w))];
        delta -=
            intData[max(0,min(x+xMinInt, h))*(w+1)
                  + max(0,min(y+yMaxInt+1, w))];
        delta +=
            intData[max(0,min(x+xMinInt, h))*(w+1)
                  + max(0,min(y+yMaxInt  , w))];

        delta *= (y+yMaxInt >= 1 and y+yMaxInt < w);
        *tmpArray = delta;
    }
}

__global__ void yMinDeltaIntegralReplicateKernel(
    const float *intData, const int intDataStrideChannel, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMin) {
 
    int id = NUM_THREADS * blockIdx.x + threadIdx.x;
    tmpArray += id; // tmpArray now points to our output pixel

    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int windowIdx = id % nWindows; id /= nWindows;

    // `id` is now is now the current global input plane number
    intData  += id * intDataStrideChannel;

    const int globalWindowIdx = (id % nInputPlane) * nWindows + windowIdx; id /= nInputPlane;
    const int & batchIdx = id;

    if (batchIdx < batchSize) {

        const int xMinInt = (int)ceil(xMin[globalWindowIdx]-1);
        const int yMinInt = (int)ceil(yMin[globalWindowIdx]-1);
        const int xMaxInt = (int)floor(xMax[globalWindowIdx]);
        // const int yMaxInt = (int)floor(yMax[globalWindowIdx]);

        float delta = 0;

        delta += 
            intData[max(0,min(x+xMaxInt, h))*(w+1)
                  + max(0,min(y+yMinInt  , w))];
        delta -=
            intData[max(0,min(x+xMaxInt, h))*(w+1)
                  + max(0,min(y+yMinInt-1, w))];
        delta -=
            intData[max(0,min(x+xMinInt, h))*(w+1)
                  + max(0,min(y+yMinInt  , w))];
        delta +=
            intData[max(0,min(x+xMinInt, h))*(w+1)
                  + max(0,min(y+yMinInt-1, w))];

        delta *= (y+yMinInt >= 1 and y+yMinInt < w);
        *tmpArray = -delta;
    }
}

extern "C"
void backwardReplicateCuda(THCState *state, const int paramId,
    const float *intData, const int intDataStrideChannel, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMin, const float *yMax,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        THError("NYI");
        // strided::backwardFracCuda(
        //     intData, tmpArray, nWindows, h, w,
        //     xMin, xMax, yMin, yMax, inData, inDataStrideRow,
        //     strideH, strideW);
        return;
    }

    const int threadsNeeded = batchSize * nInputPlane * nWindows * h * w;
    const int numBlocks = (threadsNeeded + NUM_THREADS - 1) / NUM_THREADS;

    switch (paramId) {
    case 0:
        xMinDeltaIntegralReplicateKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMin, yMin, yMax); break;
    case 1:
        xMaxDeltaIntegralReplicateKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMax, yMin, yMax); break;
    case 2:
        yMinDeltaIntegralReplicateKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMin, xMax, yMin); break;
    case 3:
        yMaxDeltaIntegralReplicateKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMin, xMax, yMax); break;
    }
    THCudaCheck(cudaGetLastError());
}

/************************ Other stuff ************************/

__global__ void dirtyFixWindowsKernel(
    float *xMin, float *xMax, float *yMin, float *yMax,
    const int size, const float h, const float w, const float minWidth) {

    int idx = NUM_THREADS * blockIdx.x + threadIdx.x;

    if (idx < 2*size) {
        float paramMin, paramMax;

        if (idx < size) {
            paramMin = max(-h+1, min(h-1, xMin[idx]));
            paramMax = max(-h+1, min(h-1, xMax[idx]));

            if (paramMin + minWidth - 0.99 > paramMax) {
                const float mean = 0.5 * (paramMin + paramMax);
                paramMin = mean - 0.5 * (minWidth - 0.9);
                paramMax = mean + 0.5 * (minWidth - 0.9);
            }

            xMin[idx] = paramMin;
            xMax[idx] = paramMax;
        } else {
            idx -= size;
            paramMin = max(-w+1, min(w-1, yMin[idx]));
            paramMax = max(-w+1, min(w-1, yMax[idx]));

            if (paramMin + minWidth - 0.99 > paramMax) {
                const float mean = 0.5 * (paramMin + paramMax);
                paramMin = mean - 0.5 * (minWidth - 0.9);
                paramMax = mean + 0.5 * (minWidth - 0.9);
            }

            yMin[idx] = paramMin;
            yMax[idx] = paramMax;
        }
    }
}

extern "C"
void dirtyFixWindows(THCState *state,
    float *xMin, float *xMax, float *yMin, float *yMax,
    int size, int h, int w, float minWidth) {

    dim3 dimBlock(NUM_THREADS);
    dim3 dimGrid((2*size + dimBlock.x - 1) / dimBlock.x);

    dirtyFixWindowsKernel <<<dimGrid, dimBlock, 0, THCState_getCurrentStream(state)>>> (
        xMin, xMax, yMin, yMax, size, (float)h, (float)w, minWidth);
    THCudaCheck(cudaGetLastError());
}
