#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "inplace/transpose.h"

#define BLOCK_SIZE 32
#define BLOCK_CHANNELS (1024 / (BLOCK_SIZE * BLOCK_SIZE))

using std::max;
using std::min;
using std::floor;
using std::ceil;

cublasHandle_t cublasHandle;

extern "C"
void _initCublasHandle() {
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
    }
    cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_HOST);

    // TODO: at shutdown, `cublasDestroy(handle);`
}

/************************ Integral image computation ************************/

__global__ void accumulateRowsKernel(
    float *input, float *output, int channels, int totalRows, int w);
__global__ void accumulateColsKernel(
    float *input, float *output, int channels, int h, int w);
__global__ void accumulateColsInplaceKernel(
    float *input, int channels, int h, int w);
__global__ void accumulateColsInplaceTransposedKernel(
    float *input, int channels, int h, int w);

extern "C"
void integralImageCuda(float *input, float *output, int channels, int h, int w) {
    int blockSize1D, gridSize1D;

    int totalCols = channels * w;
    blockSize1D = BLOCK_SIZE * BLOCK_SIZE;
    gridSize1D = (totalCols + blockSize1D - 1) / blockSize1D;
    accumulateColsKernel <<<gridSize1D, blockSize1D>>> (input, output, channels, h, w);

    inplace::transpose(true, output, channels * (h+1), w+1);

    int totalRows = channels * h;
    blockSize1D = BLOCK_SIZE * BLOCK_SIZE;
    gridSize1D = (totalRows + blockSize1D - 1) / blockSize1D;
    accumulateColsInplaceTransposedKernel <<<gridSize1D, blockSize1D>>> (output, channels, h, w);

    inplace::transpose(true, output, w+1, channels * (h+1));

    // const float alpha = 1.0, beta = 0.0;
    // A: (channels * h) x (w)
    // cublasSgeam(
    //     cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, w, channels * h,
    //     &alpha, input, channels * h,
    //     &beta, input, channels * h,
    //     output, w);
}

__global__ void accumulateRowsKernel(
    float *input, float *output, int channels, int h, int w) {
    // view multichannel image as a multiline single-channel image
    int globalRowIdx = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;

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

__global__ void accumulateColsKernel(float *input, float *output, int channels, int h, int w) {
    // global column index (of all `channels * w` columns in this image)
    int colIdx = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (colIdx < channels * w) {
        // jump to current channel
        input  += (colIdx / w) * h * w;
        output += (colIdx / w) * (h+1) * (w+1);
        colIdx %= w; // switch to local column index,
        ++colIdx;    // it's 1-indexed because first output column is always zero

        output[colIdx] = 0; // first element of every column is always zero
        double sum = 0;

        for (int i = 1; i <= h; ++i) {
            sum += input[(i-1) * w + colIdx - 1];
            output[i * (w+1) + colIdx] = sum;
        }
    }
}

__global__ void accumulateColsInplaceTransposedKernel(float *input, int channels, int h, int w) {
    // in-place.
    // input is a `(w+1) x channels * (h+1)` array

    // global column index (of all `channels * w` columns in this image)
    int colIdx = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (colIdx < channels * h) {
        // need to zero the (0,0) corner of the output separately >:(
        input[(colIdx / h) * (h+1)] = 0;

        colIdx += colIdx / h + 1; // make `colIdx` the (h+1)-array indexer

        input[colIdx] = 0; // first element of every column is always zero

        double sum = 0;

        for (int i = 1; i <= w; ++i) {
            float *currentElement = &input[i * channels * (h+1) + colIdx];
            sum += *currentElement;
            *currentElement = sum;
        }
    }
}

__global__ void accumulateColsInplaceKernel(float *input, int channels, int h, int w) {
    // in-place.
    // input is already a `channels * (h+1) x (w+1)` array

    // global column index (of all `channels * w` columns in this image)
    int colIdx = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (colIdx < channels * w) {
        input += (colIdx / w) * (h+1) * (w+1); // jump to current channel
        colIdx %= w; // switch to local column index,
        ++colIdx;    // it's 1-indexed because first output column is always zero

        input[colIdx] = 0; // first element of every column is always zero
        double sum = 0;

        for (int i = 1; i <= h; ++i) {
            float *currentElement = &input[i * (w+1) + colIdx];
            sum += *currentElement;
            *currentElement = sum;
        }
    }
}

/************************ Box filters ************************/

__global__ void forwardKernel(
    float *intData, float *outData, int h, int w, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax, float *areaCoeff) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    int z = BLOCK_CHANNELS * blockIdx.z + threadIdx.z;

    if (x < h and y < w and z < nWindows) {

        // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
        // `integral()` behavior. Namely, I(x,0) and I(0,y) are
        // always 0 (so it's a C-style array sum).

        // However, when computing sums, we subtract values at indices 
        // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
        // and yMin, and thus finally they are not affected.

        int t = max(0, min(x+(int)ceil (xMin[z]  ), h) );
        int b = max(0, min(x+(int)floor(xMax[z]+1), h) );
        int l = max(0, min(y+(int)ceil (yMin[z]  ), w) );
        int r = max(0, min(y+(int)floor(yMax[z]+1), w) );

        outData[z*w*h + x*w + y] = areaCoeff[z] *
            ( intData[b*(w+1) + r]
            - intData[t*(w+1) + r]
            - intData[b*(w+1) + l]
            + intData[t*(w+1) + l]);
    }
}

__global__ void forwardNoNormKernel(
    float *intData, float *outData, int h, int w, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    int z = BLOCK_CHANNELS * blockIdx.z + threadIdx.z;

    if (x < h and y < w and z < nWindows) {

        // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
        // `integral()` behavior. Namely, I(x,0) and I(0,y) are
        // always 0 (so it's a C-style array sum).

        // However, when computing sums, we subtract values at points 
        // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
        // and yMin, and thus finally they are not affected.

        // xMinCurr = (int)ceil(xMin[z])
        int t = max(0, min(x+(int)ceil(xMin[z]), h-1) );
        // xMaxCurr = (int)floor(xMax[z])+1
        int b = max(1, min(x+(int)floor(xMax[z])+1, h)   );
        // yMinCurr = (int)ceil(yMin[z])
        int l = max(0, min(y+(int)ceil(yMin[z]), w-1) );
        // yMaxCurr = (int)floor(yMax[z])+1
        int r = max(1, min(y+(int)floor(yMax[z])+1, w)   );

        outData[z*w*h + x*w + y] = 
            ( intData[b*(w+1) + r]
            - intData[t*(w+1) + r]
            - intData[b*(w+1) + l]
            + intData[t*(w+1) + l]);
    }
}

__global__ void forwardNoNormFracKernel(
    float *intData, float *outData, int h, int w, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStride) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    int z = BLOCK_CHANNELS * blockIdx.z + threadIdx.z;

    if (x < h and y < w and z < nWindows) {

        // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
        // `integral()` behavior. Namely, I(x,0) and I(0,y) are
        // always 0 (so it's a C-style array sum).

        // However, when computing sums, we subtract values at points 
        // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
        // and yMin, and thus finally they are not affected.

        int   xMinCurr = (int)ceil(xMin[z]);
        float xMinCurrFrac = (float)xMinCurr - xMin[z];
        int   yMinCurr = (int)ceil(yMin[z]);
        float yMinCurrFrac = (float)yMinCurr - yMin[z];

        int   xMaxCurr = (int)floor(xMax[z]);
        float xMaxCurrFrac = xMax[z] - (float)xMaxCurr;
        ++xMaxCurr;
        int   yMaxCurr = (int)floor(yMax[z]);
        float yMaxCurrFrac = yMax[z] - (float)yMaxCurr;
        ++yMaxCurr;

        int t = max(0, min(x+xMinCurr, h-1) );
        int b = max(1, min(x+xMaxCurr, h)   );
        int l = max(0, min(y+yMinCurr, w-1) );
        int r = max(1, min(y+yMaxCurr, w)   );

        outData[z*w*h + x*w + y] = 
            ( intData[b*(w+1) + r]
            - intData[t*(w+1) + r]
            - intData[b*(w+1) + l]
            + intData[t*(w+1) + l])

        // -- xMax border
        +(intData[max(1,min(x+xMaxCurr+1,h))*(w+1) 
            + max(1,min(y+yMaxCurr,w))]
        - intData[max(1,min(x+xMaxCurr,h))*(w+1)
            + max(1,min(y+yMaxCurr,w))]
        - intData[max(1,min(x+xMaxCurr+1,h))*(w+1)
            + max(0,min(y+yMinCurr,w-1))]
        + intData[max(1,min(x+xMaxCurr,h))*(w+1)
            + max(0,min(y+yMinCurr,w-1))]
        ) * xMaxCurrFrac

        // -- yMax border
        +(intData[max(1,min(x+xMaxCurr,h))*(w+1) 
            + max(1,min(y+yMaxCurr+1,w))]
        - intData[max(1,min(x+xMaxCurr,h))*(w+1)
            + max(1,min(y+yMaxCurr,w))]
        - intData[max(0,min(x+xMinCurr,h-1))*(w+1)
            + max(1,min(y+yMaxCurr+1,w))]
        + intData[max(0,min(x+xMinCurr,h-1))*(w+1)
            + max(1,min(y+yMaxCurr,w))]
        ) * yMaxCurrFrac

        // -- xMin border
        +(intData[max(0,min(x+xMinCurr,h-1))*(w+1) 
            + max(1,min(y+yMaxCurr,w))]
        - intData[max(0,min(x+xMinCurr-1,h-1))*(w+1)
            + max(1,min(y+yMaxCurr,w))]
        - intData[max(0,min(x+xMinCurr,h-1))*(w+1)
            + max(0,min(y+yMinCurr,w-1))]
        + intData[max(0,min(x+xMinCurr-1,h-1))*(w+1)
            + max(0,min(y+yMinCurr,w-1))]
        ) * xMinCurrFrac

        // -- yMin border
        +(intData[max(1,min(x+xMaxCurr,h))*(w+1) 
            + max(0,min(y+yMinCurr,w-1))]
        - intData[max(1,min(x+xMaxCurr,h))*(w+1)
            + max(0,min(y+yMinCurr-1,w-1))]
        - intData[max(0,min(x+xMinCurr,h-1))*(w+1)
            + max(0,min(y+yMinCurr,w-1))]
        + intData[max(0,min(x+xMinCurr,h-1))*(w+1)
            + max(0,min(y+yMinCurr-1,w-1))]
        ) * yMinCurrFrac

        // -- corner pixels
        + xMaxCurrFrac*yMaxCurrFrac * (
               (x+xMaxCurr > h-1 or
                y+yMaxCurr > w-1 or
                x+xMaxCurr < 0   or
                y+yMaxCurr < 0) ? 0 :
               inData[(x+xMaxCurr)*inDataStride + (y+yMaxCurr)])

        + xMinCurrFrac*yMaxCurrFrac * (
               (x+xMinCurr-1 > h-1 or
                y+yMaxCurr   > w-1 or
                x+xMinCurr-1 < 0   or
                y+yMaxCurr   < 0) ? 0 :
               inData[(x+xMinCurr-1)*inDataStride + (y+yMaxCurr)])

        + xMaxCurrFrac*yMinCurrFrac * (
               (x+xMaxCurr   > h-1 or
                y+yMinCurr-1 > w-1 or
                x+xMaxCurr   < 0   or
                y+yMinCurr-1 < 0) ? 0 :
               inData[(x+xMaxCurr)*inDataStride + (y+yMinCurr-1)])

        + xMinCurrFrac*yMinCurrFrac * (
               (x+xMinCurr-1 > h-1 or
                y+yMinCurr-1 > w-1 or
                x+xMinCurr-1 < 0   or
                y+yMinCurr-1 < 0) ? 0 :
               inData[(x+xMinCurr-1)*inDataStride + (y+yMinCurr-1)]);
    }
}

extern "C" {

void forwardCuda(
    float *intData, int h, int w, int nWindows, float *outData,
    float *xMin, float *xMax, float *yMin, float *yMax, float *areaCoeff) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_CHANNELS);
    dim3 dimGrid((h + dimBlock.x - 1) / dimBlock.x, (w + dimBlock.y - 1) / dimBlock.y, (nWindows + dimBlock.z - 1) / dimBlock.z);

    forwardKernel <<<dimGrid, dimBlock>>> (intData, outData, h, w, nWindows, xMin, xMax, yMin, yMax, areaCoeff);
}

void forwardCudaNoNorm(
    float *intData, int h, int w, int nWindows, float *outData,
    float *xMin, float *xMax, float *yMin, float *yMax) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_CHANNELS);
    dim3 dimGrid(
        (h + dimBlock.x - 1) / dimBlock.x, 
        (w + dimBlock.y - 1) / dimBlock.y, 
        (nWindows + dimBlock.z - 1) / dimBlock.z);

    forwardNoNormKernel <<<dimGrid, dimBlock>>> (intData, outData, h, w, nWindows, xMin, xMax, yMin, yMax);
}

void forwardCudaNoNormFrac(
    float *intData, int h, int w, int nWindows, float *outData,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStride) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_CHANNELS);
    dim3 dimGrid(
        (h + dimBlock.x - 1) / dimBlock.x, 
        (w + dimBlock.y - 1) / dimBlock.y, 
        (nWindows + dimBlock.z - 1) / dimBlock.z);

    forwardNoNormFracKernel <<<dimGrid, dimBlock>>> (
        intData, outData, h, w, nWindows, 
        xMin, xMax, yMin, yMax, inData, inDataStride);
}

// TODO templates for frac/non-frac

__global__ void xMaxDeltaIntegral(
    float *intData, float *tmpArray, int h, int w,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x + 1;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y + 1;

    if (x <= h and y <= w) {

        tmpArray[(x-1)*w + (y-1)] = 
            ( intData[max(0,min(x+xMaxCurr  , h))*(w+1) 
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[max(0,min(x+xMaxCurr-1, h))*(w+1) 
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[max(0,min(x+xMaxCurr  , h))*(w+1)
                + max(0,min(y+yMinCurr, w))]
            + intData[max(0,min(x+xMaxCurr-1, h))*(w+1)
                + max(0,min(y+yMinCurr, w))]);
    }
}

__global__ void xMinDeltaIntegral(
    float *intData, float *tmpArray, int h, int w,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x + 1;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y + 1;

    if (x <= h and y <= w) {

        tmpArray[(x-1)*w + (y-1)] = 
            ( intData[max(0,min(x+xMinCurr-1, h))*(w+1) 
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[min(h,max(x+xMinCurr  , 0))*(w+1)
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[max(0,min(x+xMinCurr-1, h))*(w+1)
                + max(0,min(y+yMinCurr  , w))]
            + intData[min(h,max(x+xMinCurr  , 0))*(w+1)
                + max(0,min(y+yMinCurr  , w))]);
    }
}

__global__ void yMaxDeltaIntegral(
    float *intData, float *tmpArray, int h, int w,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x + 1;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y + 1;

    if (x <= h and y <= w) {

        tmpArray[(x-1)*w + (y-1)] = 
            ( intData[max(0,min(x+xMaxCurr-1, h))*(w+1) 
                + max(0,min(y+yMaxCurr,w))]
            - intData[max(0,min(x+xMaxCurr-1, h))*(w+1)
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[max(0,min(x+xMinCurr,h))*(w+1)
                + max(0,min(y+yMaxCurr,w))]
            + intData[max(0,min(x+xMinCurr,h))*(w+1)
                + max(0,min(y+yMaxCurr-1, w))]);
    }
}

__global__ void yMinDeltaIntegral(
    float *intData, float *tmpArray, int h, int w,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x + 1;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y + 1;

    if (x <= h and y <= w) {

        tmpArray[(x-1)*w + (y-1)] = 
            ( intData[max(0,min(x+xMaxCurr-1, h))*(w+1) 
                + max(0,min(y+yMinCurr-1,w))]
            - intData[max(0,min(x+xMaxCurr-1, h))*(w+1)
                + min(w,max(y+yMinCurr, 0))]
            - intData[max(0,min(x+xMinCurr  , h))*(w+1)
                + max(0,min(y+yMinCurr-1,w))]
            + intData[max(0,min(x+xMinCurr  , h))*(w+1)
                + min(w,max(y+yMinCurr, 0))]);
    }
}

void backwardCudaSingle(
    float *intData, float *gradOutData, float *tmpArray, float *tmpArraySum, int h, int w, 
    float *deltas, int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((h + dimBlock.x - 1) / dimBlock.x, (w + dimBlock.y - 1) / dimBlock.y);

    dim3 dimBlock1D(BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimGrid1D((h*w + dimBlock1D.x - 1) / dimBlock1D.x);

    // xMaxDelta
    xMaxDeltaIntegral <<<dimGrid, dimBlock>>> (intData, tmpArray, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr);
    cublasSdot(cublasHandle, h*w, tmpArray, 1, gradOutData, 1, deltas+1);

    // xMinDelta
    xMinDeltaIntegral <<<dimGrid, dimBlock>>> (intData, tmpArray, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr);
    cublasSdot(cublasHandle, h*w, tmpArray, 1, gradOutData, 1, deltas+0);

    // yMaxDelta
    yMaxDeltaIntegral <<<dimGrid, dimBlock>>> (intData, tmpArray, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr);
    cublasSdot(cublasHandle, h*w, tmpArray, 1, gradOutData, 1, deltas+3);

    // yMinDelta
    yMinDeltaIntegral <<<dimGrid, dimBlock>>> (intData, tmpArray, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr);
    cublasSdot(cublasHandle, h*w, tmpArray, 1, gradOutData, 1, deltas+2);

    cudaDeviceSynchronize();
}

__global__ void xMaxDeltaIntegralFrac(
    float *intData, float *tmpArray, int h, int w,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float yMinCurrFrac, float yMaxCurrFrac,
    float *inData, int inDataStride) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x + 1;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y + 1;

    if (x <= h and y <= w) {

        float blCorner = (x+xMaxCurr > h or y+yMinCurr > w or
                          x+xMaxCurr < 1 or y+yMinCurr < 1) ? 0 :
                          inData[(x+xMaxCurr-1)*inDataStride + (y+yMinCurr-1)];
        float brCorner = (x+xMaxCurr > h or y+yMaxCurr > w or
                          x+xMaxCurr < 1 or y+yMaxCurr < 1) ? 0 :
                          inData[(x+xMaxCurr-1)*inDataStride + (y+yMaxCurr-1)];

        tmpArray[(x-1)*w + (y-1)] = 
            ( intData[max(0,min(x+xMaxCurr  , h))*(w+1) 
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[max(0,min(x+xMaxCurr-1, h))*(w+1) 
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[max(0,min(x+xMaxCurr  , h))*(w+1)
                + max(0,min(y+yMinCurr, w))]
            + intData[max(0,min(x+xMaxCurr-1, h))*(w+1)
                + max(0,min(y+yMinCurr, w))]
            + brCorner * yMaxCurrFrac
            + blCorner * yMinCurrFrac);
    }
}

__global__ void xMinDeltaIntegralFrac(
    float *intData, float *tmpArray, int h, int w,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float yMinCurrFrac, float yMaxCurrFrac,
    float *inData, int inDataStride) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x + 1;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y + 1;

    if (x <= h and y <= w) {

        float tlCorner = (x+xMinCurr > h or y+yMinCurr > w or 
                          x+xMinCurr < 1 or y+yMinCurr < 1) ? 0 :
                          inData[(x+xMinCurr-1)*inDataStride + (y+yMinCurr-1)];
        float trCorner = (x+xMinCurr > h or y+yMaxCurr > w or
                          x+xMinCurr < 1 or y+yMaxCurr < 1) ? 0 :
                          inData[(x+xMinCurr-1)*inDataStride + (y+yMaxCurr-1)];

        tmpArray[(x-1)*w + (y-1)] = 
            ( intData[max(0,min(x+xMinCurr-1, h))*(w+1) 
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[min(h,max(x+xMinCurr  , 0))*(w+1)
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[max(0,min(x+xMinCurr-1, h))*(w+1)
                + max(0,min(y+yMinCurr  , w))]
            + intData[min(h,max(x+xMinCurr  , 0))*(w+1)
                + max(0,min(y+yMinCurr  , w))]
            + trCorner * yMaxCurrFrac
            + tlCorner * yMinCurrFrac);
    }
}

__global__ void yMaxDeltaIntegralFrac(
    float *intData, float *tmpArray, int h, int w,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float xMinCurrFrac, float xMaxCurrFrac,
    float *inData, int inDataStride) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x + 1;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y + 1;

    if (x <= h and y <= w) {

        float trCorner = (x+xMinCurr > h or y+yMaxCurr > w or
                          x+xMinCurr < 1 or y+yMaxCurr < 1) ? 0 :
                          inData[(x+xMinCurr-1)*inDataStride + (y+yMaxCurr-1)];
        float brCorner = (x+xMaxCurr > h or y+yMaxCurr > w or
                          x+xMaxCurr < 1 or y+yMaxCurr < 1) ? 0 :
                          inData[(x+xMaxCurr-1)*inDataStride + (y+yMaxCurr-1)];

        tmpArray[(x-1)*w + (y-1)] = 
            ( intData[max(0,min(x+xMaxCurr-1, h))*(w+1) 
                + max(0,min(y+yMaxCurr,w))]
            - intData[max(0,min(x+xMaxCurr-1, h))*(w+1)
                + max(0,min(y+yMaxCurr-1, w))]
            - intData[max(0,min(x+xMinCurr,h))*(w+1)
                + max(0,min(y+yMaxCurr,w))]
            + intData[max(0,min(x+xMinCurr,h))*(w+1)
                + max(0,min(y+yMaxCurr-1, w))]
            + trCorner * xMinCurrFrac
            + brCorner * xMaxCurrFrac);
    }
}

__global__ void yMinDeltaIntegralFrac(
    float *intData, float *tmpArray, int h, int w,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float xMinCurrFrac, float xMaxCurrFrac,
    float *inData, int inDataStride) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x + 1;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y + 1;

    if (x <= h and y <= w) {

        float tlCorner = (x+xMinCurr > h or y+yMinCurr > w or 
                          x+xMinCurr < 1 or y+yMinCurr < 1) ? 0 :
                          inData[(x+xMinCurr-1)*inDataStride + (y+yMinCurr-1)];
        float blCorner = (x+xMaxCurr > h or y+yMinCurr > w or
                          x+xMaxCurr < 1 or y+yMinCurr < 1) ? 0 :
                          inData[(x+xMaxCurr-1)*inDataStride + (y+yMinCurr-1)];

        tmpArray[(x-1)*w + (y-1)] = 
            ( intData[max(0,min(x+xMaxCurr-1, h))*(w+1) 
                + max(0,min(y+yMinCurr-1,w))]
            - intData[max(0,min(x+xMaxCurr-1, h))*(w+1)
                + min(w,max(y+yMinCurr, 0))]
            - intData[max(0,min(x+xMinCurr  , h))*(w+1)
                + max(0,min(y+yMinCurr-1,w))]
            + intData[max(0,min(x+xMinCurr  , h))*(w+1)
                + min(w,max(y+yMinCurr, 0))]
            + tlCorner * xMinCurrFrac
            + blCorner * xMaxCurrFrac);
    }
}

// TODO need general kernels instead of `single`
void backwardCudaSingleFrac(
    float *intData, float *gradOutData, float *tmpArray, float *tmpArraySum, int h, int w, 
    float *deltas, int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float xMinCurrFrac, float xMaxCurrFrac, float yMinCurrFrac, float yMaxCurrFrac,
    float *inData, int inDataStride) {

    // `deltas` is a device pointer

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((h + dimBlock.x - 1) / dimBlock.x, (w + dimBlock.y - 1) / dimBlock.y);

    dim3 dimBlock1D(BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimGrid1D((h*w + dimBlock1D.x - 1) / dimBlock1D.x);

    // xMaxDelta
    xMaxDeltaIntegralFrac <<<dimGrid, dimBlock>>> (
        intData, tmpArray, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
        yMinCurrFrac, yMaxCurrFrac, inData, inDataStride);
    cublasSdot(cublasHandle, h*w, tmpArray, 1, gradOutData, 1, deltas+1);
    
    // xMinDelta
    xMinDeltaIntegralFrac <<<dimGrid, dimBlock>>> (
        intData, tmpArray, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
        yMinCurrFrac, yMaxCurrFrac, inData, inDataStride);
    cublasSdot(cublasHandle, h*w, tmpArray, 1, gradOutData, 1, deltas+0);

    // yMaxDelta
    yMaxDeltaIntegralFrac <<<dimGrid, dimBlock>>> (
        intData, tmpArray, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
        xMinCurrFrac, xMaxCurrFrac, inData, inDataStride);
    cublasSdot(cublasHandle, h*w, tmpArray, 1, gradOutData, 1, deltas+3);

    // yMinDelta
    yMinDeltaIntegralFrac <<<dimGrid, dimBlock>>> (
        intData, tmpArray, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr,
        xMinCurrFrac, xMaxCurrFrac, inData, inDataStride);
    cublasSdot(cublasHandle, h*w, tmpArray, 1, gradOutData, 1, deltas+2);

    cudaDeviceSynchronize();
}

} // extern "C"
