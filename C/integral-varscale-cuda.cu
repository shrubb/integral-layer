#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>

// #include "integral-strided-cuda.hpp"

#define BLOCK_SIZE 32
#define BLOCK_CHANNELS (1024 / (BLOCK_SIZE * BLOCK_SIZE))

using std::max;
using std::min;
using std::floor;
using std::ceil;

cublasHandle_t cublasHandle;
float *CUDA_ZERO_FLOAT, *CUDA_ONE_FLOAT; // for cublas in device pointer mode

extern "C"
void _initCublasHandleVarScale() {
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

/************************ updateOutput ************************/

// TODO
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
    float *intData, int intDataStrideChannel, float *outData,
    int h, int w, int nInputPlane, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStrideRow, int inDataStrideChannel,
    const float *scaleData) {

    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int y = id % w; id /= w;
    const int x = id % h; id /= h;
    const int & z = id;

    const int inPlaneIdx = z / nWindows;

    intData += intDataStrideChannel * inPlaneIdx;
    inData  +=  inDataStrideChannel * inPlaneIdx;

    if (x < h and y < w and z < nInputPlane*nWindows) {

        // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
        // `integral()` behavior. Namely, I(x,0) and I(0,y) are
        // always 0 (so it's a C-style array sum).

        // However, when computing sums, we subtract values at points 
        // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
        // and yMin, and thus finally they are not affected.

        const float scale = scaleData[x*w + y];

        const int   xMinCurr = (int)ceil(xMin[z] * scale);
        const float xMinCurrFrac = (float)xMinCurr - xMin[z] * scale;
        const int   yMinCurr = (int)ceil(yMin[z] * scale);
        const float yMinCurrFrac = (float)yMinCurr - yMin[z] * scale;

        const float xMaxCurrFrac = xMax[z] * scale - floor(xMax[z] * scale);
        const int   xMaxCurr = (int)floor(xMax[z] * scale) + 1;
        const float yMaxCurrFrac = yMax[z] * scale - floor(yMax[z] * scale);
        const int   yMaxCurr = (int)floor(yMax[z] * scale) + 1;

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

// TODO
void forwardNoNormReplicateCuda(
    float *intData, int intDataStrideChannel, float *outData,
    int h, int w, int nInputPlane, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        // TODO
        int a = *(int*)0;

        // strided::forwardNoNormReplicateCuda(
        //     intData, intDataStrideChannel, outData,
        //     h, w, nInputPlane, nWindows,
        //     xMin, xMax, yMin, yMax,
        //     strideH, strideW);
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

void forwardNoNormReplicateFracVarScaleCuda(
    float *intData, int intDataStrideChannel, float *outData,
    int h, int w, int nInputPlane, int nWindows,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStrideRow, int inDataStrideChannel,
    const int strideH, const int strideW, const float *scaleData) {

    if (strideH != 1 or strideW != 1) {
        // TODO
        int a = *(int*)0;

        // strided::forwardNoNormReplicateFracCuda(
        //     intData, intDataStrideChannel, outData,
        //     h, w, nInputPlane, nWindows,
        //     xMin, xMax, yMin, yMax,
        //     inData, inDataStrideRow, inDataStrideChannel,
        //     strideH, strideW);
        return;
    }

    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimGrid((nInputPlane*nWindows*h*w + dimBlock.x - 1) / dimBlock.x);

    forwardNoNormReplicateFracKernel <<<dimGrid, dimBlock>>> (
        intData, intDataStrideChannel, outData,
        h, w, nInputPlane, nWindows, 
        xMin, xMax, yMin, yMax,
        inData, inDataStrideRow, inDataStrideChannel, scaleData);
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
        // TODO
        int a = *(int*)0;

        // strided::updateGradInputCuda(
        //     gradOutputIntData, gradInputData, h, w, nWindows,
        //     xMin, xMax, yMin, yMax, strideH, strideW);
        // return;
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
        // TODO
        int a = *(int*)0;
        
        // strided::updateGradInputFracCuda(
        //     gradOutputIntData, gradInputData, h, w, nWindows,
        //     xMin, xMax, yMin, yMax,
        //     gradOutputData, gradOutputStrideRow, gradOutputStrideChannel,
        //     strideH, strideW);
        // return;
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

void backwardFracCuda(
    float *intData, float *tmpArray,
    int nWindows, int h, int w,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStrideRow,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        // TODO
        int a = *(int*)0;
        
        // strided::backwardFracCuda(
        //     intData, tmpArray, nWindows, h, w,
        //     xMin, xMax, yMin, yMax, inData, inDataStrideRow,
        //     strideH, strideW);
        // return;
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
        // TODO
        int a = *(int*)0;
        
        // strided::backwardCuda(
        //     intData, tmpArray, nWindows, h, w,
        //     xMin, xMax, yMin, yMax, strideH, strideW);
        // return;
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

/************************ Other stuff ************************/

__global__ void dirtyFixWindowsKernel(
    float *xMin, float *xMax, float *yMin, float *yMax,
    const int size, const float h, const float w, const float minWidth) {

    int idx = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;

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

void dirtyFixWindows(
    float *xMin, float *xMax, float *yMin, float *yMax,
    int size, int h, int w, float minWidth) {

    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimGrid((2*size + dimBlock.x - 1) / dimBlock.x);

    dirtyFixWindowsKernel <<<dimGrid, dimBlock>>> (
        xMin, xMax, yMin, yMax, size, (float)h, (float)w, minWidth);
}

} // extern "C"
