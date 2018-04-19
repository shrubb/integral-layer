#include <stdio.h>
#include <cmath>
#include <algorithm>

#include <cuda_runtime.h>

#include <THC/THC.h>

#define BLOCK_SIZE 32
#define BLOCK_CHANNELS (1024 / (BLOCK_SIZE * BLOCK_SIZE))

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

namespace strided {

/************************ updateOutput ************************/

// Divides x by y (y > 0), rounds towards minus infinity 
__device__ inline
int divFloor(const int x, const int y) {
    return x >= 0 ? x / y : (x - y + 1) / y;
}

// Divides x by y (y > 0), rounds towards minus infinity, returns positive remainder
__device__ inline
int modFloor(const int x, const int y) {
    return x >= 0 ? x % y : (y + x % y);
}

__global__ void forwardNoNormReplicateKernel(
    const float *intData, const int intDataStrideChannel, float *outData,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *const xMin, const float *const xMax,
    const float *const yMin, const float *const yMax,
    const int strideH, const int strideW) {

    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    outData += id; // outData now points to our output pixel

    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;

    const int y = id % wOut; id /= wOut;
    const int x = id % hOut; id /= hOut;
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

        const int t = max(0, min(x*strideH+(int) ceil(xMin[globalWindowIdx])  , h-1) );
        const int b = max(1, min(x*strideH+(int)floor(xMax[globalWindowIdx])+1, h  ) );
        const int l = max(0, min(y*strideW+(int) ceil(yMin[globalWindowIdx])  , w-1) );
        const int r = max(1, min(y*strideW+(int)floor(yMax[globalWindowIdx])+1, w  ) );

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
    const float *inData, const int inDataStrideRow, const int inDataStrideChannel,
    const int strideH, const int strideW) {

    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    outData += id; // outData now points to our output pixel

    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;

    const int y = id % wOut; id /= wOut;
    const int x = id % hOut; id /= hOut;
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

        const int t = max(0, min(x*strideH+xMinCurr, h-1) );
        const int b = max(1, min(x*strideH+xMaxCurr, h)   );
        const int l = max(0, min(y*strideW+yMinCurr, w-1) );
        const int r = max(1, min(y*strideW+yMaxCurr, w)   );

        double outValue = 0;

        outValue += intData[b*(w+1) + r];
        outValue -= intData[t*(w+1) + r];
        outValue -= intData[b*(w+1) + l];
        outValue += intData[t*(w+1) + l];

        // TODO: tAdv, bAdv, lAdv, rAdv
        // -- xMax border
        outValue +=
            ( intData[max(1,min(x*strideH+xMaxCurr+1,h))*(w+1) 
                + max(1,min(y*strideW+yMaxCurr,w))]
            - intData[max(1,min(x*strideH+xMaxCurr,h))*(w+1)
                + max(1,min(y*strideW+yMaxCurr,w))]
            - intData[max(1,min(x*strideH+xMaxCurr+1,h))*(w+1)
                + max(0,min(y*strideW+yMinCurr,w-1))]
            + intData[max(1,min(x*strideH+xMaxCurr,h))*(w+1)
                + max(0,min(y*strideW+yMinCurr,w-1))]
            ) * xMaxCurrFrac;

        // -- yMax border
        outValue +=
            ( intData[max(1,min(x*strideH+xMaxCurr,h))*(w+1) 
                + max(1,min(y*strideW+yMaxCurr+1,w))]
            - intData[max(1,min(x*strideH+xMaxCurr,h))*(w+1)
                + max(1,min(y*strideW+yMaxCurr,w))]
            - intData[max(0,min(x*strideH+xMinCurr,h-1))*(w+1)
                + max(1,min(y*strideW+yMaxCurr+1,w))]
            + intData[max(0,min(x*strideH+xMinCurr,h-1))*(w+1)
                + max(1,min(y*strideW+yMaxCurr,w))]
            ) * yMaxCurrFrac;

        // -- xMin border
        outValue +=
            ( intData[max(0,min(x*strideH+xMinCurr,h-1))*(w+1) 
                + max(1,min(y*strideW+yMaxCurr,w))]
            - intData[max(0,min(x*strideH+xMinCurr-1,h-1))*(w+1)
                + max(1,min(y*strideW+yMaxCurr,w))]
            - intData[max(0,min(x*strideH+xMinCurr,h-1))*(w+1)
                + max(0,min(y*strideW+yMinCurr,w-1))]
            + intData[max(0,min(x*strideH+xMinCurr-1,h-1))*(w+1)
                + max(0,min(y*strideW+yMinCurr,w-1))]
            ) * xMinCurrFrac;

        // -- yMin border
        outValue +=
            ( intData[max(1,min(x*strideH+xMaxCurr,h))*(w+1) 
                + max(0,min(y*strideW+yMinCurr,w-1))]
            - intData[max(1,min(x*strideH+xMaxCurr,h))*(w+1)
                + max(0,min(y*strideW+yMinCurr-1,w-1))]
            - intData[max(0,min(x*strideH+xMinCurr,h-1))*(w+1)
                + max(0,min(y*strideW+yMinCurr,w-1))]
            + intData[max(0,min(x*strideH+xMinCurr,h-1))*(w+1)
                + max(0,min(y*strideW+yMinCurr-1,w-1))]
            ) * yMinCurrFrac;

        // -- corner pixels
        outValue += 
            xMaxCurrFrac*yMaxCurrFrac * (
               (x*strideH+xMaxCurr >  h-1 or
                y*strideW+yMaxCurr >  w-1 or
                x*strideH+xMaxCurr <= 0   or
                y*strideW+yMaxCurr <= 0) ? 0 : 
                    inData[(x*strideH+xMaxCurr)*inDataStrideRow + (y*strideW+yMaxCurr)]);

        outValue +=
            xMinCurrFrac*yMaxCurrFrac * (
               (x*strideH+xMinCurr-1 >= h-1 or
                y*strideW+yMaxCurr   >  w-1 or
                x*strideH+xMinCurr-1 <  0   or
                y*strideW+yMaxCurr   <= 0) ? 0 : 
                    inData[(x*strideH+xMinCurr-1)*inDataStrideRow + (y*strideW+yMaxCurr)]);

        outValue +=
            xMaxCurrFrac*yMinCurrFrac * (
               (x*strideH+xMaxCurr   >  h-1 or
                y*strideW+yMinCurr-1 >= w-1 or
                x*strideH+xMaxCurr   <= 0   or
                y*strideW+yMinCurr-1 <  0) ? 0 : 
                    inData[(x*strideH+xMaxCurr)*inDataStrideRow + (y*strideW+yMinCurr-1)]);

        outValue +=
            xMinCurrFrac*yMinCurrFrac * (
               (x*strideH+xMinCurr-1 >= h-1 or
                y*strideW+yMinCurr-1 >= w-1 or
                x*strideH+xMinCurr-1 <  0   or
                y*strideW+yMinCurr-1 <  0) ? 0 : 
                    inData[(x*strideH+xMinCurr-1)*inDataStrideRow + (y*strideW+yMinCurr-1)]);
        
        *outData = outValue;
    }
}

void forwardNoNormReplicateCuda(THCState *state,
    const float *intData, const int intDataStrideChannel, float *outData,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMin, const float *yMax,
    const int strideH, const int strideW) {

    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;

    const int NUM_THREADS = BLOCK_SIZE * BLOCK_SIZE;
    const int threadsNeeded = batchSize * nInputPlane * nWindows * hOut * wOut;
    const int numBlocks = (threadsNeeded + NUM_THREADS - 1) / NUM_THREADS;
    
    forwardNoNormReplicateKernel
        <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
        intData, intDataStrideChannel, outData,
        batchSize, nInputPlane, nWindows, h, w,
        xMin, xMax, yMin, yMax,
        strideH, strideW);
    THCudaCheck(cudaGetLastError());
}

void forwardNoNormReplicateFracCuda(THCState *state,
    const float *intData, const int intDataStrideChannel, float *outData,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMin, const float *yMax,
    const float *inData, const int inDataStrideRow, const int inDataStrideChannel,
    const int strideH, const int strideW) {

    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;

    const int NUM_THREADS = BLOCK_SIZE * BLOCK_SIZE;
    const int threadsNeeded = batchSize * nInputPlane * nWindows * hOut * wOut;
    const int numBlocks = (threadsNeeded + NUM_THREADS - 1) / NUM_THREADS;

    forwardNoNormReplicateFracKernel
        <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
        intData, intDataStrideChannel, outData,
        batchSize, nInputPlane, nWindows, h, w,
        xMin, xMax, yMin, yMax,
        inData, inDataStrideRow, inDataStrideChannel,
        strideH, strideW);
    THCudaCheck(cudaGetLastError());
}

/************************ updateGradInput ************************/

__global__ void updateGradInputReplicatePlanewiseKernel(
    const float *gradOutputIntData, float * const gradInputData,
    const int h, const int w, const int nWindows,
    const float * const xMin, const float * const xMax,
    const float * const yMin, const float * const yMax,
    const int strideH, const int strideW) {

    const int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;

    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;

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

            const int t = max(0, min(divFloor(x+xMinCurr + strideH - 1, strideH)    , hOut) );
            const int b = max(0, min(divFloor(x+xMaxCurr - 1          , strideH) + 1, hOut) );
            const int l = max(0, min(divFloor(y+yMinCurr + strideW - 1, strideW)    , wOut) );
            const int r = max(0, min(divFloor(y+yMaxCurr - 1          , strideW) + 1, wOut) );

            outValue += gradOutputIntData[b*(wOut+1) + r];
            outValue -= gradOutputIntData[t*(wOut+1) + r];
            outValue -= gradOutputIntData[b*(wOut+1) + l];
            outValue += gradOutputIntData[t*(wOut+1) + l];

            // go to the next channel
            gradOutputIntData += (hOut+1)*(wOut+1);
        }

        gradInputData[x*w + y] = outValue;
    }
}

__global__ void updateGradInputReplicatePlanewiseFracKernel(
    const float *gradOutputIntData, float * const gradInputData,
    const int h, const int w, const int nWindows,
    const float * const xMin, const float * const xMax,
    const float * const yMin, const float * const yMax,
    const float *gradOutputData, const int gradOutputStrideRow,
    const int gradOutputStrideChannel,
    const int strideH, const int strideW) {

    const int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;

    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;

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

            const int t = max(0, min(divFloor(x+xMinCurr + strideH - 1, strideH)    , hOut) );
            const int b = max(0, min(divFloor(x+xMaxCurr - 1          , strideH) + 1, hOut) );
            const int l = max(0, min(divFloor(y+yMinCurr + strideW - 1, strideW)    , wOut) );
            const int r = max(0, min(divFloor(y+yMaxCurr - 1          , strideW) + 1, wOut) );

            const int tAdv = modFloor(x+xMinCurr-1, strideH) == 0 and x+xMinCurr-1 <  h ? max(0, min(t-1, hOut)) : t;
            const int bAdv = modFloor(x+xMaxCurr  , strideH) == 0 and x+xMaxCurr   >= 0 ? max(0, min(b+1, hOut)) : b;
            const int lAdv = modFloor(y+yMinCurr-1, strideW) == 0 and y+yMinCurr-1 <  w ? max(0, min(l-1, wOut)) : l;
            const int rAdv = modFloor(y+yMaxCurr  , strideW) == 0 and y+yMaxCurr   >= 0 ? max(0, min(r+1, wOut)) : r;

            // TODO: 1D grid
            outValue += gradOutputIntData[b*(wOut+1) + r];
            outValue -= gradOutputIntData[t*(wOut+1) + r];
            outValue -= gradOutputIntData[b*(wOut+1) + l];
            outValue += gradOutputIntData[t*(wOut+1) + l];

            // -- xMax border
            outValue +=
                ( gradOutputIntData[bAdv*(wOut+1) + r]
                - gradOutputIntData[b   *(wOut+1) + r]
                - gradOutputIntData[bAdv*(wOut+1) + l]
                + gradOutputIntData[b   *(wOut+1) + l]
                ) * xMaxCurrFrac;

            // -- yMax border
            outValue +=
                ( gradOutputIntData[b*(wOut+1) + rAdv]
                - gradOutputIntData[b*(wOut+1) + r   ]
                - gradOutputIntData[t*(wOut+1) + rAdv]
                + gradOutputIntData[t*(wOut+1) + r   ]
                ) * yMaxCurrFrac;

            // -- xMin border
            outValue +=
                ( gradOutputIntData[t   *(wOut+1) + r]
                - gradOutputIntData[tAdv*(wOut+1) + r]
                - gradOutputIntData[t   *(wOut+1) + l]
                + gradOutputIntData[tAdv*(wOut+1) + l]
                ) * xMinCurrFrac;

            // -- yMin border
            outValue +=
                ( gradOutputIntData[b*(wOut+1) + l   ]
                - gradOutputIntData[b*(wOut+1) + lAdv]
                - gradOutputIntData[t*(wOut+1) + l   ]
                + gradOutputIntData[t*(wOut+1) + lAdv]
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
            gradOutputIntData += (hOut+1)*(wOut+1);
            gradOutputData += gradOutputStrideChannel;
        }

        gradInputData[x*w + y] = outValue;
    }
}

void updateGradInputReplicatePlanewiseCuda(
    const float *gradOutputIntData, float * const gradInputData,
    const int h, const int w, const int nWindows,
    const float * const xMin, const float * const xMax,
    const float * const yMin, const float * const yMax,
    const int strideH, const int strideW) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_CHANNELS);
    dim3 dimGrid(
        (h + dimBlock.x - 1) / dimBlock.x, 
        (w + dimBlock.y - 1) / dimBlock.y);

    updateGradInputReplicatePlanewiseKernel <<<dimGrid, dimBlock>>> (
        gradOutputIntData, gradInputData,
        h, w, nWindows,
        xMin, xMax, yMin, yMax,
        strideH, strideW);
}

void updateGradInputReplicatePlanewiseFracCuda(
    const float *gradOutputIntData, float * const gradInputData,
    const int h, const int w, const int nWindows,
    const float *xMin, const float *xMax, const float *yMin, float *yMax,
    const float *gradOutputData, const int gradOutputStrideRow,
    const int gradOutputStrideChannel,
    const int strideH, const int strideW) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_CHANNELS);
    dim3 dimGrid(
        (h + dimBlock.x - 1) / dimBlock.x, 
        (w + dimBlock.y - 1) / dimBlock.y);

    updateGradInputReplicatePlanewiseFracKernel <<<dimGrid, dimBlock>>> (
        gradOutputIntData, gradInputData,
        h, w, nWindows,
        xMin, xMax, yMin, yMax,
        gradOutputData, gradOutputStrideRow, gradOutputStrideChannel,
        strideH, strideW);
}

/************************ accGradParameters ************************/

__global__ void xMaxDeltaIntegralReplicateFracKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMax, const float *yMin, const float *yMax,
    const float *inData, const int inDataStrideRow,
    const int strideH, const int strideW) {
 
    // TODO: use block dim instead
    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;
    
    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int yOut = id % wOut; id /= wOut; // 0-indexed
    const int xOut = id % hOut; id /= hOut; // 0-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and xOut < hOut and yOut < wOut) {

        const int x = xOut*strideH + 1;
        const int y = yOut*strideW + 1;

        tmpArray += windowIdx * hOut * wOut;

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
        tmpArray[xOut*wOut + yOut] = delta;
    }
}

__global__ void xMinDeltaIntegralReplicateFracKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *yMin, const float *yMax,
    const float *inData, const int inDataStrideRow,
    const int strideH, const int strideW) {
 
    // TODO: use block dim instead
    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;
    
    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int yOut = id % wOut; id /= wOut; // 0-indexed
    const int xOut = id % hOut; id /= hOut; // 0-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and xOut < hOut and yOut < wOut) {

        const int x = xOut*strideH + 1;
        const int y = yOut*strideW + 1;

        tmpArray += windowIdx * hOut * wOut;

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
        tmpArray[xOut*wOut + yOut] *= -delta;
    }
}

__global__ void yMaxDeltaIntegralReplicateFracKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMax,
    const float *inData, const int inDataStrideRow,
    const int strideH, const int strideW) {
 
    // TODO: use block dim instead
    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;
    
    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int yOut = id % wOut; id /= wOut; // 0-indexed
    const int xOut = id % hOut; id /= hOut; // 0-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and xOut < hOut and yOut < wOut) {

        const int x = xOut*strideH + 1;
        const int y = yOut*strideW + 1;

        tmpArray += windowIdx * hOut * wOut;

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
        tmpArray[xOut*wOut + yOut] = delta;
    }
}

__global__ void yMinDeltaIntegralReplicateFracKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMin,
    const float *inData, const int inDataStrideRow,
    const int strideH, const int strideW) {
 
    // TODO: use block dim instead
    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;
    
    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int yOut = id % wOut; id /= wOut; // 0-indexed
    const int xOut = id % hOut; id /= hOut; // 0-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and xOut < hOut and yOut < wOut) {

        const int x = xOut*strideH + 1;
        const int y = yOut*strideW + 1;

        tmpArray += windowIdx * hOut * wOut;

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
        tmpArray[xOut*wOut + yOut] *= -delta;
    }
}

void backwardReplicateFracCuda(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float * const xMin, const float * const xMax,
    const float * const yMin, const float * const yMax,
    const float *inData, const int inDataStrideRow,
    const int strideH, const int strideW) {

    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;

    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimGrid((nWindows * hOut * wOut + dimBlock.x - 1) / dimBlock.x);

    xMaxDeltaIntegralReplicateFracKernel <<<dimGrid, dimBlock>>> (
        intData, tmpArray + 0*nWindows*hOut*wOut, nWindows, h, w,
        xMax, yMin, yMax, inData, inDataStrideRow, strideH, strideW);

    xMinDeltaIntegralReplicateFracKernel <<<dimGrid, dimBlock>>> (
        intData, tmpArray + 1*nWindows*hOut*wOut, nWindows, h, w,
        xMin, yMin, yMax, inData, inDataStrideRow, strideH, strideW);

    yMaxDeltaIntegralReplicateFracKernel <<<dimGrid, dimBlock>>> (
        intData, tmpArray + 2*nWindows*hOut*wOut, nWindows, h, w,
        xMin, xMax, yMax, inData, inDataStrideRow, strideH, strideW);

    yMinDeltaIntegralReplicateFracKernel <<<dimGrid, dimBlock>>> (
        intData, tmpArray + 3*nWindows*hOut*wOut, nWindows, h, w,
        xMin, xMax, yMin, inData, inDataStrideRow, strideH, strideW);
}

__global__ void xMaxDeltaIntegralReplicateKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMax, const float *yMin, const float *yMax,
    const int strideH, const int strideW) {
 
    // TODO: use block dim instead
    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;

    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int yOut = id % wOut; id /= wOut; // 0-indexed
    const int xOut = id % hOut; id /= hOut; // 0-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and xOut < hOut and yOut < wOut) {

        const int x = xOut*strideH + 1;
        const int y = yOut*strideW + 1;

        tmpArray += windowIdx * hOut * wOut;

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
        tmpArray[xOut*wOut + yOut] = delta;
    }
}

__global__ void xMinDeltaIntegralReplicateKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *yMin, const float *yMax,
    const int strideH, const int strideW) {
 
    // TODO: use block dim instead
    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;
    
    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int yOut = id % wOut; id /= wOut; // 0-indexed
    const int xOut = id % hOut; id /= hOut; // 0-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and xOut < hOut and yOut < wOut) {

        const int x = xOut*strideH + 1;
        const int y = yOut*strideW + 1;

        tmpArray += windowIdx * hOut * wOut;

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
        tmpArray[xOut*wOut + yOut] *= -delta;
    }
}

__global__ void yMaxDeltaIntegralReplicateKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMax,
    const int strideH, const int strideW) {
 
    // TODO: use block dim instead
    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;
    
    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int yOut = id % wOut; id /= wOut; // 0-indexed
    const int xOut = id % hOut; id /= hOut; // 0-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and xOut < hOut and yOut < wOut) {

        const int x = xOut*strideH + 1;
        const int y = yOut*strideW + 1;

        tmpArray += windowIdx * hOut * wOut;

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
        tmpArray[xOut*wOut + yOut] = delta;
    }
}

__global__ void yMinDeltaIntegralReplicateKernel(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMin,
    const int strideH, const int strideW) {
 
    // TODO: use block dim instead
    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;
    
    int id = BLOCK_SIZE * BLOCK_SIZE * blockIdx.x + threadIdx.x;
    const int yOut = id % wOut; id /= wOut; // 0-indexed
    const int xOut = id % hOut; id /= hOut; // 0-indexed
    const int & windowIdx = id;

    if (windowIdx < nWindows and xOut < hOut and yOut < wOut) {

        const int x = xOut*strideH + 1;
        const int y = yOut*strideW + 1;

        tmpArray += windowIdx * hOut * wOut;

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
        tmpArray[xOut*wOut + yOut] *= -delta;
    }
}

void backwardReplicateCuda(
    const float *intData, float *tmpArray,
    const int nWindows, const int h, const int w,
    const float * const xMin, const float * const xMax,
    const float * const yMin, const float * const yMax,
    const int strideH, const int strideW) {

    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;

    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimGrid((nWindows * hOut * wOut + dimBlock.x - 1) / dimBlock.x);

    xMaxDeltaIntegralReplicateKernel <<<dimGrid, dimBlock>>> (
        intData, tmpArray + 0*nWindows*hOut*wOut,
        nWindows, h, w, xMax, yMin, yMax, strideH, strideW);

    xMinDeltaIntegralReplicateKernel <<<dimGrid, dimBlock>>> (
        intData, tmpArray + 1*nWindows*hOut*wOut,
        nWindows, h, w, xMin, yMin, yMax, strideH, strideW);

    yMaxDeltaIntegralReplicateKernel <<<dimGrid, dimBlock>>> (
        intData, tmpArray + 2*nWindows*hOut*wOut,
        nWindows, h, w, xMin, xMax, yMax, strideH, strideW);

    yMinDeltaIntegralReplicateKernel <<<dimGrid, dimBlock>>> (
        intData, tmpArray + 3*nWindows*hOut*wOut,
        nWindows, h, w, xMin, xMax, yMin, strideH, strideW);
}

} // namespace
