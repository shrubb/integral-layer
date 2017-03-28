#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <cmath>

#define BLOCK_SIZE 32

using std::max;
using std::min;
using std::floor;
using std::ceil;

__global__ void forwardKernelSingle(
    float *intData, float *outData, int h, int w, int xMinCurr, 
    int xMaxCurr, int yMinCurr, int yMaxCurr, float areaCoeff) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;

    if (x < h and y < w) {
        int t = max(0, min(x+xMinCurr, h) );
        int b = max(0, min(x+xMaxCurr, h) );
        int l = max(0, min(y+yMinCurr, w) );
        int r = max(0, min(y+yMaxCurr, w) );

        outData[x*w + y] = areaCoeff *
            ( intData[b*(w+1) + r]
            - intData[t*(w+1) + r]
            - intData[b*(w+1) + l]
            + intData[t*(w+1) + l]);
    }
}

__global__ void forwardKernel(
    float *intData, float *outData, int h, int w, 
    float *xMin, float *xMax, float *yMin, float *yMax, float *areaCoeff) {

    int x = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    int y = BLOCK_SIZE * blockIdx.y + threadIdx.y;
    // below we also mean: // int nWindow = blockIdx.z;

    if (x < h and y < w) {

        // Must add 1 to xMax/yMax/xMin/yMin due to OpenCV's
        // `integral()` behavior. Namely, I(x,0) and I(0,y) are
        // always 0 (so it's a C-style array sum).

        // However, when computing sums, we subtract values at indices 
        // like y+yMin-1 and x+xMin-1, so we also SUBTRACT 1 from xMin
        // and yMin, and thus finally they are not affected.

        int t = max(0, min(x+(int)ceil (xMin[blockIdx.z]  ), h) );
        int b = max(0, min(x+(int)floor(xMax[blockIdx.z]+1), h) );
        int l = max(0, min(y+(int)ceil (yMin[blockIdx.z]  ), w) );
        int r = max(0, min(y+(int)floor(yMax[blockIdx.z]+1), w) );

        outData[blockIdx.z*w*h + x*w + y] = areaCoeff[blockIdx.z] *
            ( intData[b*(w+1) + r]
            - intData[t*(w+1) + r]
            - intData[b*(w+1) + l]
            + intData[t*(w+1) + l]);
    }
}

extern "C" {

void forwardCudaSingle(
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr, float areaCoeff) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((h + dimBlock.x - 1) / dimBlock.x, (w + dimBlock.y - 1) / dimBlock.y);

    forwardKernelSingle <<<dimGrid, dimBlock>>> (intData, outData, h, w, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr, areaCoeff);
}

void forwardCuda(
    float *intData, int h, int w, int nWindows, float *outData,
    float *xMin, float *xMax, float *yMin, float *yMax, float *areaCoeff) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1024 / (BLOCK_SIZE * BLOCK_SIZE));
    dim3 dimGrid((h + dimBlock.x - 1) / dimBlock.x, (w + dimBlock.y - 1) / dimBlock.y, nWindows);

    forwardKernel <<<dimGrid, dimBlock>>> (intData, outData, h, w, xMin, xMax, yMin, yMax, areaCoeff);
}

void backwardCudaSingle(
    float *intData, float *gradOutData, int h, int w, float *deltas,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    float & xMaxDelta = deltas[1];
    float & xMinDelta = deltas[0];
    float & yMaxDelta = deltas[3];
    float & yMinDelta = deltas[2];

    for (int x = 1; x <= h; ++x) {
        for (int y = 1; y <= w; ++y) {
            
            int tClip = max(x+xMinCurr, 0);
            int bClip = min(x+xMaxCurr, h);
            int lClip = max(y+yMinCurr, 0);
            int rClip = min(y+yMaxCurr, w);

            xMaxDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[max(0,min(x+xMaxCurr+1,h))*(w+1) + max(0,rClip)]
                - intData[max(0,bClip)*(w+1) + max(0,rClip)]
                - intData[max(0,min(x+xMaxCurr+1,h))*(w+1)
                    + max(0,min(y+yMinCurr-1,w))]
                + intData[max(0,bClip)*(w+1)
                    + max(0,min(y+yMinCurr-1,w))] );
            
            xMinDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[max(0,min(x+xMinCurr-1,h))*(w+1) 
                    + max(0,rClip)]
                - intData[min(h,tClip)*(w+1)
                    + max(0,rClip)]
                - intData[max(0,min(x+xMinCurr-1,h))*(w+1)
                    + max(0,min(y+yMinCurr-1,w))]
                + intData[min(h,tClip)*(w+1)
                    + max(0,min(y+yMinCurr-1,w))] );
            
            yMaxDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[max(0,bClip)*(w+1) 
                    + max(0,min(y+yMaxCurr+1,w))]
                - intData[max(0,bClip)*(w+1)
                    + max(0,rClip)]
                - intData[max(0,min(x+xMinCurr-1,h))*(w+1)
                    + max(0,min(y+yMaxCurr+1,w))]
                + intData[max(0,min(x+xMinCurr-1,h))*(w+1)
                    + max(0,rClip)] );
            
            yMinDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[max(0,bClip)*(w+1) 
                    + max(0,min(y+yMinCurr-1,w))]
                - intData[max(0,bClip)*(w+1)
                    + min(w,lClip)]
                - intData[max(0,min(x+xMinCurr-1,h))*(w+1)
                    + max(0,min(y+yMinCurr-1,w))]
                + intData[max(0,min(x+xMinCurr-1,h))*(w+1)
                    + min(w,lClip)] );
        }
    }
}

} // extern "C"
