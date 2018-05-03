#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <THC/THC.h>

#define NUM_THREADS 1024

using std::max;
using std::min;
using std::floor;
using std::ceil;

/************************ updateOutput ************************/

__global__ void forwardNoNormFracKernel(
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

        const int t = max(0, min(x+xMinCurr, h));
        const int b = max(0, min(x+xMaxCurr, h));
        const int l = max(0, min(y+yMinCurr, w));
        const int r = max(0, min(y+yMaxCurr, w));

        const int bAdv = max(0, min(x+xMaxCurr+1, h));
        const int rAdv = max(0, min(y+yMaxCurr+1, w));
        const int tAdv = max(0, min(x+xMinCurr-1, h));
        const int lAdv = max(0, min(y+yMinCurr-1, w));

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
            (x+xMaxCurr >= h) |
            (y+yMaxCurr >= w) |
            (x+xMaxCurr <  0) |
            (y+yMaxCurr <  0));
        outValue += 
            xMaxCurrFrac * yMaxCurrFrac *
            cornerIsValid *
            inData[((x+xMaxCurr) * inDataStrideRow + (y+yMaxCurr)) * cornerIsValid];

        cornerIsValid = not (
            (x+xMinCurr >  h) |
            (y+yMaxCurr >= w) |
            (x+xMinCurr <= 0) |
            (y+yMaxCurr <  0));
        outValue +=
            xMinCurrFrac * yMaxCurrFrac *
            cornerIsValid *
            inData[((x+xMinCurr-1) * inDataStrideRow + (y+yMaxCurr)) * cornerIsValid];

        cornerIsValid = not (
            (x+xMaxCurr >= h) |
            (y+yMinCurr >  w) |
            (x+xMaxCurr <  0) |
            (y+yMinCurr <= 0));
        outValue +=
            xMaxCurrFrac * yMinCurrFrac *
            cornerIsValid *
            inData[((x+xMaxCurr) * inDataStrideRow + (y+yMinCurr-1)) * cornerIsValid];

        cornerIsValid = not (
            (x+xMinCurr >  h) |
            (y+yMinCurr >  w) |
            (x+xMinCurr <= 0) |
            (y+yMinCurr <= 0));
        outValue +=
            xMinCurrFrac * yMinCurrFrac *
            cornerIsValid *
            inData[((x+xMinCurr-1) * inDataStrideRow + (y+yMinCurr-1)) * cornerIsValid];

        *outData = outValue;
    }
}

__global__ void forwardNoNormKernel(
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

        const int t = max(0, min(x+xMinCurr, h));
        const int b = max(0, min(x+xMaxCurr, h));
        const int l = max(0, min(y+yMinCurr, w));
        const int r = max(0, min(y+yMaxCurr, w));

        float outValue = 0;

        outValue += intData[b*(w+1) + r];
        outValue -= intData[t*(w+1) + r];
        outValue -= intData[b*(w+1) + l];
        outValue += intData[t*(w+1) + l];

        *outData = outValue;
    }
}

extern "C"
void forwardNoNormCuda(THCState *state,
    float *intData, int intDataStrideChannel, float *outData,
    int batchSize, int nInputPlane, int nWindows, int h, int w,
    float *xMin, float *xMax, float *yMin, float *yMax,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        THError("NYI");
        // strided::forwardNoNormReplicateCuda(state,
        //     intData, intDataStrideChannel, outData,
        //     batchSize, nInputPlane, nWindows, h, w,
        //     xMin, xMax, yMin, yMax,
        //     strideH, strideW);
        return;
    }

    const int threadsNeeded = batchSize * nInputPlane * nWindows * h * w;
    const int numBlocks = (threadsNeeded + NUM_THREADS - 1) / NUM_THREADS;

    forwardNoNormKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
        intData, intDataStrideChannel, outData,
        batchSize, nInputPlane, nWindows, h, w,
        xMin, xMax, yMin, yMax);
    THCudaCheck(cudaGetLastError());
}

extern "C"
void forwardNoNormFracCuda(THCState *state,
    float *intData, int intDataStrideChannel, float *outData,
    int batchSize, int nInputPlane, int nWindows, int h, int w,
    float *xMin, float *xMax, float *yMin, float *yMax,
    float *inData, int inDataStrideRow, int inDataStrideChannel,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        THError("NYI");
        // strided::forwardNoNormReplicateFracCuda(state,
        //     intData, intDataStrideChannel, outData,
        //     batchSize, nInputPlane, nWindows, h, w,
        //     xMin, xMax, yMin, yMax,
        //     inData, inDataStrideRow, inDataStrideChannel,
        //     strideH, strideW);
        return;
    }

    const int threadsNeeded = batchSize * nInputPlane * nWindows * h * w;
    const int numBlocks = (threadsNeeded + NUM_THREADS - 1) / NUM_THREADS;

    forwardNoNormFracKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
        intData, intDataStrideChannel, outData,
        batchSize, nInputPlane, nWindows, h, w,
        xMin, xMax, yMin, yMax,
        inData, inDataStrideRow, inDataStrideChannel);
    THCudaCheck(cudaGetLastError());
}

/************************ updateGradInput ************************/

__global__ void updateGradInputKernel(
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

__global__ void updateGradInputFracKernel(
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

        int xMinCurr, xMaxCurr, yMinCurr, yMaxCurr;

        xMinCurr = (int)ceil(-xMax[globalWindowIdx]);
        yMinCurr = (int)ceil(-yMax[globalWindowIdx]);
        const float xMinCurrFrac = (float)xMinCurr + xMax[globalWindowIdx];
        const float yMinCurrFrac = (float)yMinCurr + yMax[globalWindowIdx];

        xMaxCurr = (int)floor(-xMin[globalWindowIdx]) + 1;
        yMaxCurr = (int)floor(-yMin[globalWindowIdx]) + 1;
        const float xMaxCurrFrac = -xMin[globalWindowIdx] + 1 - xMaxCurr;
        const float yMaxCurrFrac = -yMin[globalWindowIdx] + 1 - yMaxCurr;

        const int t = max(0, min(x+xMinCurr, h));
        const int b = max(0, min(x+xMaxCurr, h));
        const int l = max(0, min(y+yMinCurr, w));
        const int r = max(0, min(y+yMaxCurr, w));

        const int tAdv = x+xMinCurr-1 <  h ? max(0, min(t-1, h)) : t;
        const int bAdv = x+xMaxCurr   >= 0 ? max(0, min(b+1, h)) : b;
        const int lAdv = y+yMinCurr-1 <  w ? max(0, min(l-1, w)) : l;
        const int rAdv = y+yMaxCurr   >= 0 ? max(0, min(r+1, w)) : r;

        float outValue = 0;

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
               (x+xMaxCurr >= h or
                y+yMaxCurr >= w or
                x+xMaxCurr <  0 or
                y+yMaxCurr <  0 or
                b == bAdv or
                r == rAdv) ? 0 : 
                gradOutputData[b*gradOutputStrideRow + r]);

        outValue +=
            xMinCurrFrac*yMaxCurrFrac * (
               (x+xMinCurr >  h or
                y+yMaxCurr >= w or
                x+xMinCurr <= 0 or
                y+yMaxCurr <  0 or
                t == tAdv or
                r == rAdv) ? 0 : 
                gradOutputData[tAdv*gradOutputStrideRow + r]);

        outValue +=
            xMaxCurrFrac*yMinCurrFrac * (
               (x+xMaxCurr >= h or
                y+yMinCurr >  w or
                x+xMaxCurr <  0 or
                y+yMinCurr <= 0 or
                b == bAdv or
                l == lAdv) ? 0 : 
                gradOutputData[b*gradOutputStrideRow + lAdv]);

        outValue +=
            xMinCurrFrac*yMinCurrFrac * (
               (x+xMinCurr >  h or
                y+yMinCurr >  w or
                x+xMinCurr <= 0 or
                y+yMinCurr <= 0 or
                t == tAdv or
                l == lAdv) ? 0 : 
                gradOutputData[tAdv*gradOutputStrideRow + lAdv]);

        *tmpArray = outValue;
    }
}

extern "C"
void updateGradInputCuda(THCState *state,
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

    updateGradInputKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
        gradOutputIntData, tmpArray,
        batchSize, nInputPlane, nWindows,
        h, w, xMin, xMax, yMin, yMax);
    THCudaCheck(cudaGetLastError());
}

extern "C"
void updateGradInputFracCuda(THCState *state,
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

    updateGradInputFracKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
        gradOutputIntData, tmpArray,
        batchSize, nInputPlane, nWindows,
        h, w, xMin, xMax, yMin, yMax,
        gradOutputData, gradOutputStrideRow, gradOutputStrideChannel);
    THCudaCheck(cudaGetLastError());
}

/************************ accGradParameters fastest *********************/

__global__ void xMaxDeltaIntegralFracKernel(
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

        int valid;

        valid = not (y+yMinInt < 1) & not (y+yMinInt > w) & not (x+xMaxInt >= h);
        const float blCorner = valid * inData[
            max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
            max(0,min(w-1,y+yMinInt-1))];
        
        valid = not (y+yMaxInt < 0) & not (y+yMaxInt >= w) & not (x+xMaxInt >= h);
        const float brCorner = valid * inData[
            max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
            max(0,min(w-1,y+yMaxInt  ))];

        float delta = 0;

        delta += brCorner * yMaxFrac;
        delta += blCorner * yMinFrac;

        delta +=
            intData[max(0,min(x+xMaxInt+1, h))*(w+1) 
                  + max(0,min(y+yMaxInt  , w))];
        delta -=
            intData[max(0,min(x+xMaxInt  , h))*(w+1) 
                  + max(0,min(y+yMaxInt  , w))];
        delta -=
            intData[max(0,min(x+xMaxInt+1, h))*(w+1)
                  + max(0,min(y+yMinInt  , w))];
        delta +=
            intData[max(0,min(x+xMaxInt  , h))*(w+1)
                  + max(0,min(y+yMinInt  , w))];

        delta *= (x+xMaxInt >= 0) & (x+xMaxInt < h);
        *tmpArray = delta;
    }
}

__global__ void xMinDeltaIntegralFracKernel(
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
    inData += id * inDataStrideChannel;

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

        valid = not (y+yMinInt < 1) & not (y+yMinInt > w) & not (x+xMinInt < 1);
        const float tlCorner = valid * inData[
            max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow + 
            max(0,min(w-1,y+yMinInt-1))];
        
        valid = not (y+yMaxInt < 0) & not (y+yMaxInt >= w) & not (x+xMinInt < 1);
        const float trCorner = valid * inData[
            max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow + 
            max(0,min(w-1,y+yMaxInt  ))];
        
        float delta = 0;

        delta += trCorner * yMaxFrac;
        delta += tlCorner * yMinFrac;

        delta += 
            intData[max(0,min(x+xMinInt  , h))*(w+1)
                  + max(0,min(y+yMaxInt  , w))];
        delta -=
            intData[max(0,min(x+xMinInt-1, h))*(w+1)
                  + max(0,min(y+yMaxInt  , w))];
        delta -=
            intData[max(0,min(x+xMinInt  , h))*(w+1)
                  + max(0,min(y+yMinInt  , w))];
        delta +=
            intData[max(0,min(x+xMinInt-1, h))*(w+1)
                  + max(0,min(y+yMinInt  , w))];

        delta *= (x+xMinInt >= 1) & (x+xMinInt <= h);
        *tmpArray = -delta;
    }
}

__global__ void yMaxDeltaIntegralFracKernel(
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

        int valid;

        valid = not (y+yMaxInt >= w) & not (x+xMinInt < 1) & not (x+xMinInt > h);
        const float trCorner = valid * inData[
            max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
            max(0,min(w-1,y+yMaxInt  ))];
       
        valid = not (y+yMaxInt >= w) & not (x+xMaxInt < 0) & not (x+xMaxInt >= h);
        const float brCorner = valid * inData[
            max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
            max(0,min(w-1,y+yMaxInt  ))];

        float delta = 0;

        delta += trCorner * xMinFrac;
        delta += brCorner * xMaxFrac;

        delta += 
            intData[max(0,min(x+xMaxInt  , h))*(w+1)
                  + max(0,min(y+yMaxInt+1, w))];
        delta -=
            intData[max(0,min(x+xMaxInt  , h))*(w+1)
                  + max(0,min(y+yMaxInt  , w))];
        delta -=
            intData[max(0,min(x+xMinInt  , h))*(w+1)
                  + max(0,min(y+yMaxInt+1, w))];
        delta +=
            intData[max(0,min(x+xMinInt  , h))*(w+1)
                  + max(0,min(y+yMaxInt  , w))];

        delta *= (y+yMaxInt >= 0) & (y+yMaxInt < w);
        *tmpArray = delta;
    }
}

__global__ void yMinDeltaIntegralFracKernel(
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

        int valid;

        valid = not (y+yMinInt < 1) & not (x+xMinInt < 1) & not (x+xMinInt > h);
        const float tlCorner = valid * inData[
            max(0,min(h-1,x+xMinInt-1)) * inDataStrideRow +
            max(0,min(w-1,y+yMinInt-1))];
        
        valid = not (y+yMinInt < 1) & not (x+xMaxInt < 0) & not (x+xMaxInt >= h);
        const float blCorner = valid * inData[
            max(0,min(h-1,x+xMaxInt  )) * inDataStrideRow +
            max(0,min(w-1,y+yMinInt-1))];
        
        float delta = 0;

        delta += tlCorner * xMinFrac;
        delta += blCorner * xMaxFrac;

        delta += 
            intData[max(0,min(x+xMaxInt  , h))*(w+1)
                  + max(0,min(y+yMinInt  , w))];
        delta -=
            intData[max(0,min(x+xMaxInt  , h))*(w+1)
                  + max(0,min(y+yMinInt-1, w))];
        delta -=
            intData[max(0,min(x+xMinInt  , h))*(w+1)
                  + max(0,min(y+yMinInt  , w))];
        delta +=
            intData[max(0,min(x+xMinInt  , h))*(w+1)
                  + max(0,min(y+yMinInt-1, w))];

        delta *= (y+yMinInt >= 1) & (y+yMinInt <= w);
        *tmpArray = -delta;
    }
}

extern "C"
void backwardFracCuda(THCState *state, const int paramId, const bool inputIsOnes,
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
        xMinDeltaIntegralFracKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMin, yMin, yMax,
            inData, inDataStrideRow, inDataStrideChannel); break;
    case 1:
        xMaxDeltaIntegralFracKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMax, yMin, yMax,
            inData, inDataStrideRow, inDataStrideChannel); break;
    case 2:
        yMinDeltaIntegralFracKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMin, xMax, yMin,
            inData, inDataStrideRow, inDataStrideChannel); break;
    case 3:
        yMaxDeltaIntegralFracKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMin, xMax, yMax,
            inData, inDataStrideRow, inDataStrideChannel); break;
    }
    THCudaCheck(cudaGetLastError());
}

__global__ void xMaxDeltaIntegralKernel(
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

__global__ void xMinDeltaIntegralKernel(
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

__global__ void yMaxDeltaIntegralKernel(
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

__global__ void yMinDeltaIntegralKernel(
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
void backwardCuda(THCState *state, const int paramId,
    const float *intData, const int intDataStrideChannel, float *tmpArray,
    const int batchSize, const int nInputPlane, const int nWindows, const int h, const int w,
    const float *xMin, const float *xMax, const float *yMin, const float *yMax,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        THError("NYI");
        // strided::backwardCuda(
        //     intData, tmpArray, nWindows, h, w,
        //     xMin, xMax, yMin, yMax, inData, inDataStrideRow,
        //     strideH, strideW);
        return;
    }

    const int threadsNeeded = batchSize * nInputPlane * nWindows * h * w;
    const int numBlocks = (threadsNeeded + NUM_THREADS - 1) / NUM_THREADS;

    switch (paramId) {
    case 0:
        xMinDeltaIntegralKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMin, yMin, yMax); break;
    case 1:
        xMaxDeltaIntegralKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMax, yMin, yMax); break;
    case 2:
        yMinDeltaIntegralKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMin, xMax, yMin); break;
    case 3:
        yMaxDeltaIntegralKernel <<<numBlocks, NUM_THREADS, 0, THCState_getCurrentStream(state)>>> (
            intData, intDataStrideChannel, tmpArray,
            batchSize, nInputPlane, nWindows, h, w,
            xMin, xMax, yMax); break;
    }
    THCudaCheck(cudaGetLastError());
}
