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
float *CUDA_ZERO_FLOAT, *CUDA_ONE_FLOAT; // for cublas in device pointer mode

extern "C"
void _initCublasHandle() {
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
    }
    cublasSetPointerMode(cublasHandle, CUBLAS_POINTER_MODE_DEVICE);
    // TODO: at shutdown, `cublasDestroy(handle);`

    // TODO: deallocate this!
    float zeroOne[] = {0, 1};
    cudaMalloc((void**)&CUDA_ZERO_FLOAT, sizeof(zeroOne));
    CUDA_ONE_FLOAT = CUDA_ZERO_FLOAT + 1;
    cudaMemcpy(CUDA_ZERO_FLOAT, zeroOne, sizeof(zeroOne), cudaMemcpyHostToDevice);
}

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
    float *input, float *output, int channels, int totalRows, int w);
__global__ void accumulateColsKernel(
    float *input, float *output, int channels, int h, int w);
__global__ void accumulateColsInplaceKernel(
    float *input, int channels, int h, int w);
__global__ void accumulateColsInplaceTransposedKernel(
    float *input, int channels, int h, int w);

extern "C"
void integralImageCuda(float *input, float *output, int channels, int h, int w, float *tmp) {
    int blockSize1D, gridSize1D;

    int totalCols = channels * w;
    blockSize1D = BLOCK_SIZE * BLOCK_SIZE;
    gridSize1D = (totalCols + blockSize1D - 1) / blockSize1D;
    accumulateColsKernel <<<gridSize1D, blockSize1D>>> (input, output, channels, h, w);

    cublasSgeam(
        cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, channels * (h+1), w+1,
        CUDA_ONE_FLOAT, output, w+1,
        CUDA_ZERO_FLOAT, tmp, channels * (h+1),
        tmp, channels * (h+1));

    int totalRows = channels * h;
    blockSize1D = BLOCK_SIZE * BLOCK_SIZE;
    gridSize1D = (totalRows + blockSize1D - 1) / blockSize1D;
    accumulateColsInplaceTransposedKernel <<<gridSize1D, blockSize1D>>> (tmp, channels, h, w);

    cublasSgeam(
        cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, w+1, channels * (h+1),
        CUDA_ONE_FLOAT, tmp, channels * (h+1),
        CUDA_ZERO_FLOAT, output, w+1,
        output, w+1);
}

extern "C"
void integralImageInplaceCuda(float *input, float *output, int channels, int h, int w) {
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
            sum += static_cast<double>(input[(i-1) * w + colIdx - 1]);
            output[i * (w+1) + colIdx] = static_cast<float>(sum);
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
            sum += static_cast<double>(*currentElement);
            *currentElement = static_cast<float>(sum);
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
            sum += static_cast<double>(*currentElement);
            *currentElement = static_cast<float>(sum);
        }
    }
}

/************************ updateOutput ************************/

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

__global__ void forwardNoNormReplicateKernel(
    float *intData, float *outData,
    int h, int w, int nInputPlane, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax) {

    const int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    const int z = BLOCK_CHANNELS * blockIdx.z + threadIdx.z;

    const int inPlaneIdx = z / nWindows;

    intData += (h+1) * (w+1) * inPlaneIdx;

    if (x < h and y < w and z < nInputPlane*nWindows) {

        // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
        // `integral()` behavior. Namely, I(x,0) and I(0,y) are
        // always 0 (so it's a C-style array sum).

        // However, when computing sums, we subtract values at points 
        // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
        // and yMin, and thus finally they are not affected.

        const int t = max(0, min(x+(int) ceil(xMin[z])  , h-1) );
        const int b = max(1, min(x+(int)floor(xMax[z])+1, h  ) );
        const int l = max(0, min(y+(int) ceil(yMin[z])  , w-1) );
        const int r = max(1, min(y+(int)floor(yMax[z])+1, w  ) );

        double outValue = 0;

        outValue += intData[b*(w+1) + r];
        outValue -= intData[t*(w+1) + r];
        outValue -= intData[b*(w+1) + l];
        outValue += intData[t*(w+1) + l];

        outData[z*w*h + x*w + y] = outValue;
    }
}

__global__ void forwardNoNormReplicateFracKernel(
    float *intData, float *outData,
    int h, int w, int nInputPlane, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStrideRow, int inDataStrideChannel) {

    const int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    const int z = BLOCK_CHANNELS * blockIdx.z + threadIdx.z;

    const int inPlaneIdx = z / nWindows;

    intData += (h+1) * (w+1) * inPlaneIdx;
    inData  += inDataStrideChannel * inPlaneIdx;

    if (x < h and y < w and z < nInputPlane*nWindows) {

        // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
        // `integral()` behavior. Namely, I(x,0) and I(0,y) are
        // always 0 (so it's a C-style array sum).

        // However, when computing sums, we subtract values at points 
        // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
        // and yMin, and thus finally they are not affected.

        const int   xMinCurr = (int)ceil(xMin[z]);
        const float xMinCurrFrac = (float)xMinCurr - xMin[z];
        const int   yMinCurr = (int)ceil(yMin[z]);
        const float yMinCurrFrac = (float)yMinCurr - yMin[z];

        const float xMaxCurrFrac = xMax[z] - floor(xMax[z]);
        const int   xMaxCurr = (int)floor(xMax[z]) + 1;
        const float yMaxCurrFrac = yMax[z] - floor(yMax[z]);
        const int   yMaxCurr = (int)floor(yMax[z]) + 1;

        const int t = max(0, min(x+xMinCurr, h-1) );
        const int b = max(1, min(x+xMaxCurr, h)   );
        const int l = max(0, min(y+yMinCurr, w-1) );
        const int r = max(1, min(y+yMaxCurr, w)   );

        double outValue = 0;

        outValue += intData[b*(w+1) + r];
        outValue -= intData[t*(w+1) + r];
        outValue -= intData[b*(w+1) + l];
        outValue += intData[t*(w+1) + l];

        // -- xMax border
        outValue +=
            ( intData[max(1,min(x+xMaxCurr+1,h))*(w+1) 
                + max(1,min(y+yMaxCurr,w))]
            - intData[max(1,min(x+xMaxCurr,h))*(w+1)
                + max(1,min(y+yMaxCurr,w))]
            - intData[max(1,min(x+xMaxCurr+1,h))*(w+1)
                + max(0,min(y+yMinCurr,w-1))]
            + intData[max(1,min(x+xMaxCurr,h))*(w+1)
                + max(0,min(y+yMinCurr,w-1))]
            ) * xMaxCurrFrac;

        // -- yMax border
        outValue +=
            ( intData[max(1,min(x+xMaxCurr,h))*(w+1) 
                + max(1,min(y+yMaxCurr+1,w))]
            - intData[max(1,min(x+xMaxCurr,h))*(w+1)
                + max(1,min(y+yMaxCurr,w))]
            - intData[max(0,min(x+xMinCurr,h-1))*(w+1)
                + max(1,min(y+yMaxCurr+1,w))]
            + intData[max(0,min(x+xMinCurr,h-1))*(w+1)
                + max(1,min(y+yMaxCurr,w))]
            ) * yMaxCurrFrac;

        // -- xMin border
        outValue +=
            ( intData[max(0,min(x+xMinCurr,h-1))*(w+1) 
                + max(1,min(y+yMaxCurr,w))]
            - intData[max(0,min(x+xMinCurr-1,h-1))*(w+1)
                + max(1,min(y+yMaxCurr,w))]
            - intData[max(0,min(x+xMinCurr,h-1))*(w+1)
                + max(0,min(y+yMinCurr,w-1))]
            + intData[max(0,min(x+xMinCurr-1,h-1))*(w+1)
                + max(0,min(y+yMinCurr,w-1))]
            ) * xMinCurrFrac;

        // -- yMin border
        outValue +=
            ( intData[max(1,min(x+xMaxCurr,h))*(w+1) 
                + max(0,min(y+yMinCurr,w-1))]
            - intData[max(1,min(x+xMaxCurr,h))*(w+1)
                + max(0,min(y+yMinCurr-1,w-1))]
            - intData[max(0,min(x+xMinCurr,h-1))*(w+1)
                + max(0,min(y+yMinCurr,w-1))]
            + intData[max(0,min(x+xMinCurr,h-1))*(w+1)
                + max(0,min(y+yMinCurr-1,w-1))]
            ) * yMinCurrFrac;

        // -- corner pixels
        outValue += 
            xMaxCurrFrac*yMaxCurrFrac * (
               (x+xMaxCurr >  h-1 or
                y+yMaxCurr >  w-1 or
                x+xMaxCurr <= 0   or
                y+yMaxCurr <= 0) ? 0 : inData[(x+xMaxCurr)*inDataStrideRow + (y+yMaxCurr)]);

        outValue +=
            xMinCurrFrac*yMaxCurrFrac * (
               (x+xMinCurr-1 >= h-1 or
                y+yMaxCurr   >  w-1 or
                x+xMinCurr-1 <  0   or
                y+yMaxCurr   <= 0) ? 0 : inData[(x+xMinCurr-1)*inDataStrideRow + (y+yMaxCurr)]);

        outValue +=
            xMaxCurrFrac*yMinCurrFrac * (
               (x+xMaxCurr   >  h-1 or
                y+yMinCurr-1 >= w-1 or
                x+xMaxCurr   <= 0   or
                y+yMinCurr-1 <  0) ? 0 : inData[(x+xMaxCurr)*inDataStrideRow + (y+yMinCurr-1)]);

        outValue +=
            xMinCurrFrac*yMinCurrFrac * (
               (x+xMinCurr-1 >= h-1 or
                y+yMinCurr-1 >= w-1 or
                x+xMinCurr-1 <  0   or
                y+yMinCurr-1 <  0) ? 0 : inData[(x+xMinCurr-1)*inDataStrideRow + (y+yMinCurr-1)]);

        outData[z*w*h + x*w + y] = outValue;
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

void forwardCudaNoNormReplicate(
    float *intData, float *outData,
    int h, int w, int nInputPlane, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_CHANNELS);
    dim3 dimGrid(
        (h + dimBlock.x - 1) / dimBlock.x, 
        (w + dimBlock.y - 1) / dimBlock.y, 
        (nInputPlane*nWindows + dimBlock.z - 1) / dimBlock.z);

    forwardNoNormReplicateKernel <<<dimGrid, dimBlock>>> (
        intData, outData, h, w, nInputPlane, nWindows,
        xMin, xMax, yMin, yMax);
}

void forwardCudaNoNormReplicateFrac(
    float *intData, float *outData,
    int h, int w, int nInputPlane, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStrideRow, int inDataStrideChannel) {

    // TODO: 1D grid
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_CHANNELS);
    dim3 dimGrid(
        (h + dimBlock.x - 1) / dimBlock.x, 
        (w + dimBlock.y - 1) / dimBlock.y, 
        (nInputPlane*nWindows + dimBlock.z - 1) / dimBlock.z);

    forwardNoNormReplicateFracKernel <<<dimGrid, dimBlock>>> (
        intData, outData, h, w, nInputPlane, nWindows, 
        xMin, xMax, yMin, yMax,
        inData, inDataStrideRow, inDataStrideChannel);
}

/************************ updateGradInput ************************/

__global__ void updateGradInputKernel(
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

__global__ void updateGradInputFracKernel(
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

            // TODO: 1D grid
            outValue += gradOutputIntData[b*(w+1) + r];
            outValue -= gradOutputIntData[t*(w+1) + r];
            outValue -= gradOutputIntData[b*(w+1) + l];
            outValue += gradOutputIntData[t*(w+1) + l];

            // -- xMax border
            outValue +=
                ( gradOutputIntData[max(0,min(x+xMaxCurr+1,h))*(w+1)
                    + max(0,min(y+yMaxCurr,w))]
                - gradOutputIntData[max(0,min(x+xMaxCurr  ,h))*(w+1)
                    + max(0,min(y+yMaxCurr,w))]
                - gradOutputIntData[max(0,min(x+xMaxCurr+1,h))*(w+1)
                    + max(0,min(y+yMinCurr,w))]
                + gradOutputIntData[max(0,min(x+xMaxCurr  ,h))*(w+1)
                    + max(0,min(y+yMinCurr,w))]
                ) * xMaxCurrFrac;

            // -- yMax border
            outValue +=
                ( gradOutputIntData[max(0,min(x+xMaxCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr+1,w))]
                - gradOutputIntData[max(0,min(x+xMaxCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr  ,w))]
                - gradOutputIntData[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr+1,w))]
                + gradOutputIntData[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr  ,w))]
                ) * yMaxCurrFrac;

            // -- xMin border
            outValue +=
                ( gradOutputIntData[max(0,min(x+xMinCurr  ,h))*(w+1)
                    + max(0,min(y+yMaxCurr,w))]
                - gradOutputIntData[max(0,min(x+xMinCurr-1,h))*(w+1)
                    + max(0,min(y+yMaxCurr,w))]
                - gradOutputIntData[max(0,min(x+xMinCurr  ,h))*(w+1)
                    + max(0,min(y+yMinCurr,w))]
                + gradOutputIntData[max(0,min(x+xMinCurr-1,h))*(w+1)
                    + max(0,min(y+yMinCurr,w))]
                ) * xMinCurrFrac;

            // -- yMin border
            outValue +=
                ( gradOutputIntData[max(0,min(x+xMaxCurr,h))*(w+1)
                    + max(0,min(y+yMinCurr  ,w))]
                - gradOutputIntData[max(0,min(x+xMaxCurr,h))*(w+1)
                    + max(0,min(y+yMinCurr-1,w))]
                - gradOutputIntData[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMinCurr  ,w))]
                + gradOutputIntData[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMinCurr-1,w))]
                ) * yMinCurrFrac;

            // -- corner pixels
            outValue += 
                xMaxCurrFrac*yMaxCurrFrac * (
                   (x+xMaxCurr > h-1 or
                    y+yMaxCurr > w-1 or
                    x+xMaxCurr < 0   or
                    y+yMaxCurr < 0) ? 0 : 
                    gradOutputData[(x+xMaxCurr)*gradOutputStrideRow + (y+yMaxCurr)]);

            outValue +=
                xMinCurrFrac*yMaxCurrFrac * (
                   (x+xMinCurr-1 > h-1 or
                    y+yMaxCurr   > w-1 or
                    x+xMinCurr-1 < 0   or
                    y+yMaxCurr   < 0) ? 0 : 
                    gradOutputData[(x+xMinCurr-1)*gradOutputStrideRow + (y+yMaxCurr)]);

            outValue +=
                xMaxCurrFrac*yMinCurrFrac * (
                   (x+xMaxCurr   > h-1 or
                    y+yMinCurr-1 > w-1 or
                    x+xMaxCurr   < 0   or
                    y+yMinCurr-1 < 0) ? 0 : 
                    gradOutputData[(x+xMaxCurr)*gradOutputStrideRow + (y+yMinCurr-1)]);

            outValue +=
                xMinCurrFrac*yMinCurrFrac * (
                   (x+xMinCurr-1 > h-1 or
                    y+yMinCurr-1 > w-1 or
                    x+xMinCurr-1 < 0   or
                    y+yMinCurr-1 < 0) ? 0 : 
                    gradOutputData[(x+xMinCurr-1)*gradOutputStrideRow + (y+yMinCurr-1)]);

            // go to the next channel
            gradOutputIntData += (h+1)*(w+1);
            gradOutputData += gradOutputStrideChannel;
        }

        gradInputData[x*w + y] = outValue;
    }
}

void updateGradInputCuda(
    float *gradOutputIntData, float *gradInputData,
    int h, int w, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_CHANNELS);
    dim3 dimGrid(
        (h + dimBlock.x - 1) / dimBlock.x, 
        (w + dimBlock.y - 1) / dimBlock.y);

    updateGradInputKernel <<<dimGrid, dimBlock>>> (
        gradOutputIntData, gradInputData,
        h, w, nWindows,
        xMin, xMax, yMin, yMax);
}

void updateGradInputCudaFrac(
    float *gradOutputIntData, float *gradInputData,
    int h, int w, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *gradOutputData, int gradOutputStrideRow, int gradOutputStrideChannel) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_CHANNELS);
    dim3 dimGrid(
        (h + dimBlock.x - 1) / dimBlock.x, 
        (w + dimBlock.y - 1) / dimBlock.y);

    updateGradInputFracKernel <<<dimGrid, dimBlock>>> (
        gradOutputIntData, gradInputData,
        h, w, nWindows,
        xMin, xMax, yMin, yMax,
        gradOutputData, gradOutputStrideRow, gradOutputStrideChannel);
}

/************************ accGradParameters ************************/

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

    // cudaDeviceSynchronize();
}

__global__ void xMaxDeltaIntegralFrac(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMin, const float *yMax,
    const float *inData, const int inDataStrideRow) {
 
    int id = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and x <= h and y <= w) {

        const int xMinInt = (int)ceil(xMin[windowIdx]-1);
        const float xMinFrac = xMinInt-xMin[windowIdx]+1;

        const int yMinInt = (int)ceil(yMin[windowIdx]-1);
        const float yMinFrac = yMinInt-yMin[windowIdx]+1;

        const int xMaxInt = (int)floor(xMax[windowIdx]);
        const float xMaxFrac = xMax[windowIdx]-xMaxInt;

        const int yMaxInt = (int)floor(yMax[windowIdx]);
        const float yMaxFrac = yMax[windowIdx]-yMaxInt;

        const float blCorner = y+yMinInt <  1 or x+xMaxInt >= h ? 0 :
                            inData[
                                max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
                                max(0,min(w-1,y+yMinInt-1))];
        const float brCorner = y+yMaxInt >= w or x+xMaxInt >= h ? 0 :
                            inData[
                                max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
                                max(0,min(w-1,y+yMaxInt  ))];

        float delta = 0;
        delta += brCorner * (y+yMaxInt[windowIdx] <  1 ? 1.0f : yMaxFrac[windowIdx]);
        delta += blCorner * (y+yMinInt[windowIdx] >= w ? 1.0f : yMinFrac[windowIdx]);

        delta += 
            intData[max(0,min(x+xMaxInt[windowIdx]+1, h))*(w+1) 
                + max(0,min(y+yMaxInt[windowIdx], w))];
        delta -=
            intData[max(0,min(x+xMaxInt[windowIdx]  , h))*(w+1) 
                + max(0,min(y+yMaxInt[windowIdx], w))];
        delta -=
            intData[max(0,min(x+xMaxInt[windowIdx]+1, h))*(w+1)
                + max(0,min(y+yMinInt[windowIdx], w))];
        delta +=
            intData[max(0,min(x+xMaxInt[windowIdx]  , h))*(w+1)
                + max(0,min(y+yMinInt[windowIdx], w))];

        tmpArray[(x-1)*w + (y-1)] *= delta;
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

void backwardCudaFrac(
    float *intData, float *tmpArray,
    int nWindows, int h, int w,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStrideRow) {

    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE * BLOCK_CHANNELS);
    dim3 dimGrid((nWindows * h * w + dimBlock.x - 1) / dimBlock.x);

    // // xMaxDelta
    // xMaxDeltaIntegralFrac <<<dimGrid, dimBlock>>> (
    //     intData, tmpArray + 0*nWindows*h*w, nWindows, h, w,
    //     xMin, xMax, yMin, yMax, inData, inDataStride);
    
    // // xMinDelta
    // xMinDeltaIntegralFrac <<<dimGrid, dimBlock>>> (
    //     intData, tmpArray + 1*nWindows*h*w, nWindows, h, w,
    //     xMin, xMax, yMin, yMax, inData, inDataStride);

    // // yMaxDelta
    // yMaxDeltaIntegralFrac <<<dimGrid, dimBlock>>> (
    //     intData, tmpArray + 2*nWindows*h*w, nWindows, h, w,
    //     xMin, xMax, yMin, yMax, inData, inDataStride);

    // // yMinDelta
    // yMinDeltaIntegralFrac <<<dimGrid, dimBlock>>> (
    //     intData, tmpArray + 3*nWindows*h*w, nWindows, h, w,
    //     xMin, xMax, yMin, yMax, inData, inDataStride);
}

} // extern "C"
