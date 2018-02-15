#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "integral-strided-cuda.hpp"

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

/*
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
*/

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
    float *intData, int intDataStrideChannel, float *outData,
    int h, int w, int nInputPlane, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax) {

    const int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    const int z = BLOCK_CHANNELS * blockIdx.z + threadIdx.z;

    const int inPlaneIdx = z / nWindows;

    intData += intDataStrideChannel * inPlaneIdx;

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
    const float *intData, const int intDataStrideChannel, float *const outData,
    const int h, const int w, const int nInputPlane, const int nWindows,
    const float *xMin, const float *xMax, const float *yMin, const float *yMax,
    const float *inData, const int inDataStrideRow, const int inDataStrideChannel) {

    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int y = id % w; id /= w;
    const int x = id % h; id /= h;
    const int windowIdx = id % nWindows; id /= nWindows;
    const int & inPlaneIdx = id;

    intData += intDataStrideChannel * inPlaneIdx;
    inData  +=  inDataStrideChannel * inPlaneIdx;

    if (x < h and y < w and windowIdx < nWindows and inPlaneIdx < nInputPlane) {

        const int rem = windowIdx % 4;

        const float xMinStretched = rem == 0 ? -h :
            xMin[inPlaneIdx*(nWindows-(nWindows-0+3) / 4) + 3*(windowIdx/4) + (rem > 0 ? (rem-1) : rem)];
        const float xMaxStretched = rem == 1 ?  h : 
            xMax[inPlaneIdx*(nWindows-(nWindows-1+3) / 4) + 3*(windowIdx/4) + (rem > 1 ? (rem-1) : rem)];
        const float yMinStretched = rem == 2 ? -w : 
            yMin[inPlaneIdx*(nWindows-(nWindows-2+3) / 4) + 3*(windowIdx/4) + (rem > 2 ? (rem-1) : rem)];
        const float yMaxStretched = rem == 3 ?  w : 
            yMax[inPlaneIdx*(nWindows-(nWindows-3+3) / 4) + 3*(windowIdx/4) + (rem > 3 ? (rem-1) : rem)];

        // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
        // `integral()` behavior. Namely, I(x,0) and I(0,y) are
        // always 0 (so it's a C-style array sum).

        // However, when computing sums, we subtract values at points 
        // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
        // and yMin, and thus finally they are not affected.

        const int   xMinCurr = (int)ceil(xMinStretched);
        const float xMinCurrFrac = (float)xMinCurr - xMinStretched;
        const int   yMinCurr = (int)ceil(yMinStretched);
        const float yMinCurrFrac = (float)yMinCurr - yMinStretched;

        const float xMaxCurrFrac = xMaxStretched - floor(xMaxStretched);
        const int   xMaxCurr = (int)floor(xMaxStretched) + 1;
        const float yMaxCurrFrac = yMaxStretched - floor(yMaxStretched);
        const int   yMaxCurr = (int)floor(yMaxStretched) + 1;

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

        outData[(inPlaneIdx*nWindows+windowIdx)*w*h + x*w + y] = outValue;
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

void forwardNoNormReplicateCuda(
    float *intData, int intDataStrideChannel, float *outData,
    int h, int w, int nInputPlane, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        strided::forwardNoNormReplicateCuda(
            intData, intDataStrideChannel, outData,
            h, w, nInputPlane, nWindows,
            xMin, xMax, yMin, yMax,
            strideH, strideW);
        return;
    }

    // TODO: 1D grid
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_CHANNELS);
    dim3 dimGrid(
        (h + dimBlock.x - 1) / dimBlock.x, 
        (w + dimBlock.y - 1) / dimBlock.y, 
        (nInputPlane*nWindows + dimBlock.z - 1) / dimBlock.z);

    forwardNoNormReplicateKernel <<<dimGrid, dimBlock>>> (
        intData, intDataStrideChannel, outData,
        h, w, nInputPlane, nWindows,
        xMin, xMax, yMin, yMax);
}

void forwardNoNormReplicateFracCuda(
    float *intData, int intDataStrideChannel, float *outData,
    int h, int w, int nInputPlane, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStrideRow, int inDataStrideChannel,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        strided::forwardNoNormReplicateFracCuda(
            intData, intDataStrideChannel, outData,
            h, w, nInputPlane, nWindows,
            xMin, xMax, yMin, yMax,
            inData, inDataStrideRow, inDataStrideChannel,
            strideH, strideW);
        return;
    }

    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimGrid((nInputPlane*nWindows*h*w + dimBlock.x - 1) / dimBlock.x);

    forwardNoNormReplicateFracKernel <<<dimGrid, dimBlock>>> (
        intData, intDataStrideChannel, outData,
        h, w, nInputPlane, nWindows, 
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
    const float *gradOutputIntData, float *const gradInputData,
    const int h, const int w, const int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    const float *gradOutputData, 
    const int gradOutputStrideRow, const int gradOutputStrideChannel) {

    const int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;

    if (x < h and y < w) {

        int xMinCurr, xMaxCurr, yMinCurr, yMaxCurr;
        double outValue = 0;

        for (int windowIdx = 0; windowIdx < nWindows; ++windowIdx) {

            const int rem = windowIdx % 4;

            const float xMinStretched = rem == 0 ? -h :
                xMin[3*(windowIdx/4) + (rem > 0 ? (rem-1) : rem)];
            const float xMaxStretched = rem == 1 ?  h : 
                xMax[3*(windowIdx/4) + (rem > 1 ? (rem-1) : rem)];
            const float yMinStretched = rem == 2 ? -w : 
                yMin[3*(windowIdx/4) + (rem > 2 ? (rem-1) : rem)];
            const float yMaxStretched = rem == 3 ?  w : 
                yMax[3*(windowIdx/4) + (rem > 3 ? (rem-1) : rem)];

            // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
            // `integral()` behavior. Namely, I(x,0) and I(0,y) are
            // always 0 (so it's a C-style array sum).

            // However, when computing sums, we subtract values at points 
            // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
            // and yMin, and thus finally they are not affected.

            xMinCurr = (int)ceil(-xMaxStretched);
            yMinCurr = (int)ceil(-yMaxStretched);
            const float xMinCurrFrac = (float)xMinCurr + xMaxStretched;
            const float yMinCurrFrac = (float)yMinCurr + yMaxStretched;

            xMaxCurr = (int)floor(-xMinStretched) + 1;
            yMaxCurr = (int)floor(-yMinStretched) + 1;
            const float xMaxCurrFrac = -xMinStretched + 1 - xMaxCurr;
            const float yMaxCurrFrac = -yMinStretched + 1 - yMaxCurr;

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

void updateGradInputCuda(
    float *gradOutputIntData, float *gradInputData,
    int h, int w, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        strided::updateGradInputCuda(
            gradOutputIntData, gradInputData, h, w, nWindows,
            xMin, xMax, yMin, yMax, strideH, strideW);
        return;
    }

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_CHANNELS);
    dim3 dimGrid(
        (h + dimBlock.x - 1) / dimBlock.x, 
        (w + dimBlock.y - 1) / dimBlock.y);

    updateGradInputKernel <<<dimGrid, dimBlock>>> (
        gradOutputIntData, gradInputData,
        h, w, nWindows,
        xMin, xMax, yMin, yMax);
}

void updateGradInputFracCuda(
    float *gradOutputIntData, float *gradInputData,
    int h, int w, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *gradOutputData, int gradOutputStrideRow, int gradOutputStrideChannel,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        strided::updateGradInputFracCuda(
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

    updateGradInputFracKernel <<<dimGrid, dimBlock>>> (
        gradOutputIntData, gradInputData,
        h, w, nWindows,
        xMin, xMax, yMin, yMax,
        gradOutputData, gradOutputStrideRow, gradOutputStrideChannel);
}

/************************ accGradParameters ************************/

__global__ void xMaxDeltaIntegralFracKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMax, const float *yMin, const float *yMax,
    const float *inData, const int inDataStrideRow) {
 
    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and x <= h and y <= w) {

        tmpArray += windowIdx * h * w;

        const int rem = windowIdx % 4;

        if (rem == 1) {
            tmpArray[(x-1)*w + (y-1)] = 0;
        } else {

            // const float xMinStretched = rem == 0 ? -h :
            //     xMin[3*(windowIdx/4) + (rem > 0 ? (rem-1) : rem)];
            const float xMaxStretched = rem == 1 ?  h : 
                xMax[3*(windowIdx/4) + (rem > 1 ? (rem-1) : rem)];
            const float yMinStretched = rem == 2 ? -w : 
                yMin[3*(windowIdx/4) + (rem > 2 ? (rem-1) : rem)];
            const float yMaxStretched = rem == 3 ?  w : 
                yMax[3*(windowIdx/4) + (rem > 3 ? (rem-1) : rem)];

            // const int xMinInt = (int)ceil(xMinStretched-1);
            // const float xMinFrac = xMinInt-xMinStretched+1;

            const int yMinInt = (int)ceil(yMinStretched-1);
            const float yMinFrac = yMinInt-yMinStretched+1;

            const int xMaxInt = (int)floor(xMaxStretched);
            // const float xMaxFrac = xMaxStretched-xMaxInt;

            const int yMaxInt = (int)floor(yMaxStretched);
            const float yMaxFrac = yMaxStretched-yMaxInt;

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
}

__global__ void xMinDeltaIntegralFracKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *yMin, const float *yMax,
    const float *inData, const int inDataStrideRow) {
 
    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and x <= h and y <= w) {

        tmpArray += windowIdx * h * w;

        const int rem = windowIdx % 4;

        if (rem == 0) {
            tmpArray[(x-1)*w + (y-1)] = 0;
        } else {
            
            const float xMinStretched = rem == 0 ? -h :
                xMin[3*(windowIdx/4) + (rem > 0 ? (rem-1) : rem)];
            // const float xMaxStretched = rem == 1 ?  h : 
            //     xMax[3*(windowIdx/4) + (rem > 1 ? (rem-1) : rem)];
            const float yMinStretched = rem == 2 ? -w : 
                yMin[3*(windowIdx/4) + (rem > 2 ? (rem-1) : rem)];
            const float yMaxStretched = rem == 3 ?  w : 
                yMax[3*(windowIdx/4) + (rem > 3 ? (rem-1) : rem)];

            const int xMinInt = (int)ceil(xMinStretched-1);
            // const float xMinFrac = xMinInt-xMinStretched+1;

            const int yMinInt = (int)ceil(yMinStretched-1);
            const float yMinFrac = yMinInt-yMinStretched+1;

            // const int xMaxInt = (int)floor(xMaxStretched);
            // const float xMaxFrac = xMaxStretched-xMaxInt;

            const int yMaxInt = (int)floor(yMaxStretched);
            const float yMaxFrac = yMaxStretched-yMaxInt;

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
}

__global__ void yMaxDeltaIntegralFracKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMax,
    const float *inData, const int inDataStrideRow) {
 
    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and x <= h and y <= w) {

        tmpArray += windowIdx * h * w;

        const int rem = windowIdx % 4;

        if (rem == 3) {
            tmpArray[(x-1)*w + (y-1)] = 0;
        } else {
            
            const float xMinStretched = rem == 0 ? -h :
                xMin[3*(windowIdx/4) + (rem > 0 ? (rem-1) : rem)];
            const float xMaxStretched = rem == 1 ?  h : 
                xMax[3*(windowIdx/4) + (rem > 1 ? (rem-1) : rem)];
            // const float yMinStretched = rem == 2 ? -w : 
            //     yMin[3*(windowIdx/4) + (rem > 2 ? (rem-1) : rem)];
            const float yMaxStretched = rem == 3 ?  w : 
                yMax[3*(windowIdx/4) + (rem > 3 ? (rem-1) : rem)];

            const int xMinInt = (int)ceil(xMinStretched-1);
            const float xMinFrac = xMinInt-xMinStretched+1;

            // const int yMinInt = (int)ceil(yMinStretched-1);
            // const float yMinFrac = yMinInt-yMinStretched+1;

            const int xMaxInt = (int)floor(xMaxStretched);
            const float xMaxFrac = xMaxStretched-xMaxInt;

            const int yMaxInt = (int)floor(yMaxStretched);
            // const float yMaxFrac = yMaxStretched-yMaxInt;

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
}

__global__ void yMinDeltaIntegralFracKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMin,
    const float *inData, const int inDataStrideRow) {
 
    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int y = id % w + 1; id /= w; // 1-indexed
    const int x = id % h + 1; id /= h; // 1-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and x <= h and y <= w) {

        tmpArray += windowIdx * h * w;

        const int rem = windowIdx % 4;

        if (rem == 2) {
            tmpArray[(x-1)*w + (y-1)] = 0;
        } else {
            
            const float xMinStretched = rem == 0 ? -h :
                xMin[3*(windowIdx/4) + (rem > 0 ? (rem-1) : rem)];
            const float xMaxStretched = rem == 1 ?  h : 
                xMax[3*(windowIdx/4) + (rem > 1 ? (rem-1) : rem)];
            const float yMinStretched = rem == 2 ? -w : 
                yMin[3*(windowIdx/4) + (rem > 2 ? (rem-1) : rem)];
            // const float yMaxStretched = rem == 3 ?  w : 
            //     yMax[3*(windowIdx/4) + (rem > 3 ? (rem-1) : rem)];

            const int xMinInt = (int)ceil(xMinStretched-1);
            const float xMinFrac = xMinInt-xMinStretched+1;

            const int yMinInt = (int)ceil(yMinStretched-1);
            // const float yMinFrac = yMinInt-yMinStretched+1;

            const int xMaxInt = (int)floor(xMaxStretched);
            const float xMaxFrac = xMaxStretched-xMaxInt;

            // const int yMaxInt = (int)floor(yMaxStretched);
            // const float yMaxFrac = yMaxStretched-yMaxInt;

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
}

void backwardFracCuda(
    float *intData, float *tmpArray,
    int nWindows, int h, int w,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStrideRow,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        strided::backwardFracCuda(
            intData, tmpArray, nWindows, h, w,
            xMin, xMax, yMin, yMax, inData, inDataStrideRow,
            strideH, strideW);
        return;
    }

    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimGrid((nWindows * h * w + dimBlock.x - 1) / dimBlock.x);

    xMaxDeltaIntegralFracKernel <<<dimGrid, dimBlock>>> (
        intData, tmpArray + 0*nWindows*h*w, nWindows, h, w,
        xMax, yMin, yMax, inData, inDataStrideRow);

    xMinDeltaIntegralFracKernel <<<dimGrid, dimBlock>>> (
        intData, tmpArray + 1*nWindows*h*w, nWindows, h, w,
        xMin, yMin, yMax, inData, inDataStrideRow);

    yMaxDeltaIntegralFracKernel <<<dimGrid, dimBlock>>> (
        intData, tmpArray + 2*nWindows*h*w, nWindows, h, w,
        xMin, xMax, yMax, inData, inDataStrideRow);

    yMinDeltaIntegralFracKernel <<<dimGrid, dimBlock>>> (
        intData, tmpArray + 3*nWindows*h*w, nWindows, h, w,
        xMin, xMax, yMin, inData, inDataStrideRow);
}

__global__ void xMaxDeltaIntegralKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMax, const float *yMin, const float *yMax) {
 
    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
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

__global__ void xMinDeltaIntegralKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *yMin, const float *yMax) {
 
    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
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

__global__ void yMaxDeltaIntegralKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMax) {
 
    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
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

__global__ void yMinDeltaIntegralKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMin) {
 
    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
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

void backwardCuda(
    float *intData, float *tmpArray,
    int nWindows, int h, int w,
    float *xMin, float *xMax, float *yMin, float *yMax,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        strided::backwardCuda(
            intData, tmpArray, nWindows, h, w,
            xMin, xMax, yMin, yMax, strideH, strideW);
        return;
    }

    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimGrid((nWindows * h * w + dimBlock.x - 1) / dimBlock.x);

    xMaxDeltaIntegralKernel <<<dimGrid, dimBlock>>> (
        intData, tmpArray + 0*nWindows*h*w,
        nWindows, h, w, xMax, yMin, yMax);

    xMinDeltaIntegralKernel <<<dimGrid, dimBlock>>> (
        intData, tmpArray + 1*nWindows*h*w,
        nWindows, h, w, xMin, yMin, yMax);

    yMaxDeltaIntegralKernel <<<dimGrid, dimBlock>>> (
        intData, tmpArray + 2*nWindows*h*w,
        nWindows, h, w, xMin, xMax, yMax);

    yMinDeltaIntegralKernel <<<dimGrid, dimBlock>>> (
        intData, tmpArray + 3*nWindows*h*w,
        nWindows, h, w, xMin, xMax, yMin);
}

__global__ void toBorderAddGradParamsKernel(
    const int nWindows,
    float *const gradXMax, float *const gradXMin,
    float *const gradYMax, float *const gradYMin,
    const float scale, const float *tmpArraySumGPU) {

    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    
    if (id < 4*nWindows) {
        int paramIdx = id / nWindows;
        float *const gradParam = (float*[]){gradXMax, gradXMin, gradYMax, gradYMin}[paramIdx];

        const int windowIdx = id % nWindows;
        const int rem = windowIdx % 4;
        
        // use streams, not this arithmetic insanity
        if ((5-rem) % 4 != paramIdx) {
            gradParam[3*(windowIdx/4) + (rem > (5-paramIdx) % 4 ? (rem-1) : rem)] += 
                scale * tmpArraySumGPU[id];   
        }
    }
}

// TODO: hey...use streams...would you be USING STREAMS INSTEAD please!
void toBorderAddGradParams(
    const int nWindows,
    float *const gradXMax, float *const gradXMin,
    float *const gradYMax, float *const gradYMin,
    const float scale, const float *const tmpArraySumGPU) {

    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimGrid((4*nWindows + dimBlock.x - 1) / dimBlock.x);

    toBorderAddGradParamsKernel <<<dimGrid, dimBlock>>> (
        nWindows, gradXMax, gradXMin, gradYMax, gradYMin, scale, tmpArraySumGPU);
}

/************************ Other stuff ************************/

__global__ void dirtyFixWindowsKernel(
    float *const xMin, float *const xMax, float *const yMin, float *const yMax,
    const int nInputPlane, const int nWindows,
    const float h, const float w, const float minWidth) {

    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;

    const bool correctingY = id >= nInputPlane*nWindows;
    if (correctingY) {
        id -= nInputPlane*nWindows;
    }

    const int windowIdx = id % nWindows; id /= nWindows;
    const int & inPlaneIdx = id;

    const int rem = windowIdx % 4;

    const int xMinIdx = inPlaneIdx*(nWindows-(nWindows-0+3) / 4) + 3*(windowIdx/4) + (rem > 0 ? (rem-1) : rem);
    const int xMaxIdx = inPlaneIdx*(nWindows-(nWindows-1+3) / 4) + 3*(windowIdx/4) + (rem > 1 ? (rem-1) : rem);
    const int yMinIdx = inPlaneIdx*(nWindows-(nWindows-2+3) / 4) + 3*(windowIdx/4) + (rem > 2 ? (rem-1) : rem);
    const int yMaxIdx = inPlaneIdx*(nWindows-(nWindows-3+3) / 4) + 3*(windowIdx/4) + (rem > 3 ? (rem-1) : rem);

    if (inPlaneIdx < nInputPlane and windowIdx < nWindows) {
        float paramMin, paramMax;

        if (not correctingY) {
            if (rem == 2 or rem == 3) {
                paramMin = max(-h+1, min(h-1, xMin[xMinIdx]));
                paramMax = max(-h+1, min(h-1, xMax[xMaxIdx]));

                if (paramMin + minWidth - 0.99 > paramMax) {
                    const float mean = 0.5 * (paramMin + paramMax);
                    paramMin = mean - 0.5 * (minWidth - 0.9);
                    paramMax = mean + 0.5 * (minWidth - 0.9);
                }

                xMin[xMinIdx] = paramMin;
                xMax[xMaxIdx] = paramMax;

            } else if (rem == 0) {
                xMax[xMaxIdx] = max(-h+1, min(h-1, xMax[xMaxIdx]));
            } else if (rem == 1) {
                xMin[xMinIdx] = max(-h+1, min(h-1, xMin[xMinIdx]));
            }
        } else {
            if (rem == 0 or rem == 1) {
                paramMin = max(-w+1, min(w-1, yMin[yMinIdx]));
                paramMax = max(-w+1, min(w-1, yMax[yMaxIdx]));

                if (paramMin + minWidth - 0.99 > paramMax) {
                    const float mean = 0.5 * (paramMin + paramMax);
                    paramMin = mean - 0.5 * (minWidth - 0.9);
                    paramMax = mean + 0.5 * (minWidth - 0.9);
                }

                yMin[yMinIdx] = paramMin;
                yMax[yMaxIdx] = paramMax;

            } else if (rem == 2) {
                yMax[yMaxIdx] = max(-w+1, min(w-1, yMax[yMaxIdx]));
            } else if (rem == 3) {
                yMin[yMinIdx] = max(-w+1, min(w-1, yMin[yMinIdx]));
            }
        }
    }
}

void dirtyFixWindows(
    float *const xMin, float *const xMax, float *const yMin, float *const yMax,
    const int nInputPlane, const int nWindows,
    const int h, const int w, const float minWidth) {

    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimGrid((2*nInputPlane*nWindows + dimBlock.x - 1) / dimBlock.x);

    dirtyFixWindowsKernel <<<dimGrid, dimBlock>>> (
        xMin, xMax, yMin, yMax, nInputPlane, nWindows, (float)h, (float)w, minWidth);
}

} // extern "C"
