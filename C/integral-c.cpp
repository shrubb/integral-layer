#include <algorithm>
#include <cmath>
#include <iostream>

#include <TH/TH.h>

using std::max;
using std::min;

int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

#include "integral-strided-c.hpp"

extern "C" {

void forward(
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr, float areaCoeff) {

    int t, b, l, r;

    // #pragma omp parallel for private(t,b,l,r)
    for (int x = 0; x < h; ++x) {
        for (int y = 0; y < w; ++y) {

            t = max(0, min(x+xMinCurr, h) );
            b = max(0, min(x+xMaxCurr, h) );
            l = max(0, min(y+yMinCurr, w) );
            r = max(0, min(y+yMaxCurr, w) );

            outData[x*w + y] = areaCoeff *
                ( intData[b*(w+1) + r]
                - intData[t*(w+1) + r]
                - intData[b*(w+1) + l]
                + intData[t*(w+1) + l]);
        }
    }
}

void backward(
    float *intData, float *gradOutData, int h, int w, float *deltas,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    float xMaxDelta = 0;
    float xMinDelta = 0;
    float yMaxDelta = 0;
    float yMinDelta = 0;

    // #pragma omp parallel for reduction(+:xMaxDelta,xMinDelta,yMaxDelta,yMinDelta)
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

    deltas[1] = xMaxDelta;
    deltas[0] = xMinDelta;
    deltas[3] = yMaxDelta;
    deltas[2] = yMinDelta;
}

void forwardNoNorm(
    const float *intData, const int h, const int w, float *outData,
    const int xMinCurr, const int xMaxCurr, const int yMinCurr, const int yMaxCurr,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        THError("NYI");
        forwardNoNormReplicateStrided(
            intData, h, w, outData, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr, strideH, strideW);
        return;
    }

    int t, b, l, r;

    // #pragma omp parallel for private(t,b,l,r)
    for (int x = 0; x < h; ++x) {
        for (int y = 0; y < w; ++y) {

            t = max(0, min(x+xMinCurr, h) );
            b = max(0, min(x+xMaxCurr, h) );
            l = max(0, min(y+yMinCurr, w) );
            r = max(0, min(y+yMaxCurr, w) );

            outData[x*w + y] = 
                ( intData[b*(w+1) + r]
                - intData[t*(w+1) + r]
                - intData[b*(w+1) + l]
                + intData[t*(w+1) + l]);
        }
    }
}

void forwardNoNormFrac(
    const float *intData, const int h, const int w, float *outData,
    const int xMinCurr, const int xMaxCurr, const int yMinCurr, const int yMaxCurr,
    const float xMinCurrFrac, const float xMaxCurrFrac, 
    const float yMinCurrFrac, const float yMaxCurrFrac,
    const float *inData, const int inDataStride,
    const int strideH, const int strideW) {

    THError("NYI");
    
    if (strideH != 1 or strideW != 1) {
        THError("NYI");
        forwardNoNormReplicateFracStrided(
            intData, h, w, outData, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr, 
            xMinCurrFrac, xMaxCurrFrac, yMinCurrFrac, yMaxCurrFrac,
            inData, inDataStride, strideH, strideW);
        return;
    }

    int t, b, l, r;

    // #pragma omp parallel for private(t,b,l,r)
    for (int x = 0; x < h; ++x) {
        for (int y = 0; y < w; ++y) {

            // note `1` / `h-1` / `w-1` because of "replicate" interpolation
            t = max(0, min(x+xMinCurr, h) );
            b = max(0, min(x+xMaxCurr, h) );
            l = max(0, min(y+yMinCurr, w) );
            r = max(0, min(y+yMaxCurr, w) );

            outData[x*w + y] = 
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
                   (x+xMaxCurr >  h-1 or
                    y+yMaxCurr >  w-1 or
                    x+xMaxCurr <= 0   or
                    y+yMaxCurr <= 0) ? 0 : inData[(x+xMaxCurr)*inDataStride + (y+yMaxCurr)])

            + xMinCurrFrac*yMaxCurrFrac * (
                   (x+xMinCurr-1 >= h-1 or
                    y+yMaxCurr   >  w-1 or
                    x+xMinCurr-1 <  0   or
                    y+yMaxCurr   <= 0) ? 0 : inData[(x+xMinCurr-1)*inDataStride + (y+yMaxCurr)])

            + xMaxCurrFrac*yMinCurrFrac * (
                   (x+xMaxCurr   >  h-1 or
                    y+yMinCurr-1 >= w-1 or
                    x+xMaxCurr   <= 0   or
                    y+yMinCurr-1 <  0) ? 0 : inData[(x+xMaxCurr)*inDataStride + (y+yMinCurr-1)])

            + xMinCurrFrac*yMinCurrFrac * (
                   (x+xMinCurr-1 >= h-1 or
                    y+yMinCurr-1 >= w-1 or
                    x+xMinCurr-1 <  0   or
                    y+yMinCurr-1 <  0) ? 0 : inData[(x+xMinCurr-1)*inDataStride + (y+yMinCurr-1)]);
        }
    }                            
}

void forwardNoNormReplicate(
    const float *intData, const int h, const int w, float *outData,
    const int xMinCurr, const int xMaxCurr, const int yMinCurr, const int yMaxCurr,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        forwardNoNormReplicateStrided(
            intData, h, w, outData, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr, strideH, strideW);
        return;
    }

    int t, b, l, r;

    // #pragma omp parallel for private(t,b,l,r)
    for (int x = 0; x < h; ++x) {
        for (int y = 0; y < w; ++y) {

            // note `1` / `h-1` / `w-1` because of "replicate" interpolation
            t = max(0, min(x+xMinCurr, h-1) );
            b = max(1, min(x+xMaxCurr, h)   );
            l = max(0, min(y+yMinCurr, w-1) );
            r = max(1, min(y+yMaxCurr, w)   );

            outData[x*w + y] = 
                ( intData[b*(w+1) + r]
                - intData[t*(w+1) + r]
                - intData[b*(w+1) + l]
                + intData[t*(w+1) + l]);
        }
    }
}

void forwardNoNormReplicateFrac(
    const float *intData, const int h, const int w, float *outData,
    const int xMinCurr, const int xMaxCurr, const int yMinCurr, const int yMaxCurr,
    const float xMinCurrFrac, const float xMaxCurrFrac, 
    const float yMinCurrFrac, const float yMaxCurrFrac,
    const float *inData, const int inDataStride,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        forwardNoNormReplicateFracStrided(
            intData, h, w, outData, xMinCurr, xMaxCurr, yMinCurr, yMaxCurr, 
            xMinCurrFrac, xMaxCurrFrac, yMinCurrFrac, yMaxCurrFrac,
            inData, inDataStride, strideH, strideW);
        return;
    }

    int t, b, l, r;

    // #pragma omp parallel for private(t,b,l,r)
    for (int x = 0; x < h; ++x) {
        for (int y = 0; y < w; ++y) {

            // note `1` / `h-1` / `w-1` because of "replicate" interpolation
            t = max(0, min(x+xMinCurr, h-1) );
            b = max(1, min(x+xMaxCurr, h)   );
            l = max(0, min(y+yMinCurr, w-1) );
            r = max(1, min(y+yMaxCurr, w)   );

            outData[x*w + y] = 
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
                   (x+xMaxCurr >  h-1 or
                    y+yMaxCurr >  w-1 or
                    x+xMaxCurr <= 0   or
                    y+yMaxCurr <= 0) ? 0 : inData[(x+xMaxCurr)*inDataStride + (y+yMaxCurr)])

            + xMinCurrFrac*yMaxCurrFrac * (
                   (x+xMinCurr-1 >= h-1 or
                    y+yMaxCurr   >  w-1 or
                    x+xMinCurr-1 <  0   or
                    y+yMaxCurr   <= 0) ? 0 : inData[(x+xMinCurr-1)*inDataStride + (y+yMaxCurr)])

            + xMaxCurrFrac*yMinCurrFrac * (
                   (x+xMaxCurr   >  h-1 or
                    y+yMinCurr-1 >= w-1 or
                    x+xMaxCurr   <= 0   or
                    y+yMinCurr-1 <  0) ? 0 : inData[(x+xMaxCurr)*inDataStride + (y+yMinCurr-1)])

            + xMinCurrFrac*yMinCurrFrac * (
                   (x+xMinCurr-1 >= h-1 or
                    y+yMinCurr-1 >= w-1 or
                    x+xMinCurr-1 <  0   or
                    y+yMinCurr-1 <  0) ? 0 : inData[(x+xMinCurr-1)*inDataStride + (y+yMinCurr-1)]);
        }
    }                            
}

void updateGradInput(
    const float *gradOutputInt, const int nWindows, const int h, const int w, float *gradInput,
    const int *xMin, const int *xMax, const int *yMin, const int *yMax,
    const float *gradOutput, const int gradOutputStride,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        updateGradInputStrided(
            gradOutputInt, nWindows, h, w, gradInput,
            xMin, xMax, yMin, yMax,
            gradOutput, gradOutputStride, strideH, strideW);
        return;
    }

    int t, b, l, r;

    for (int windowIdx = 0; windowIdx < nWindows; ++windowIdx) {

        // #pragma omp parallel for private(t,b,l,r)
        for (int x = 0; x < h; ++x) {
            for (int y = 0; y < w; ++y) {

                const int xMinCurr = (x == 0   and xMax[windowIdx] >= 0 ? 0    : xMin[windowIdx]);
                const int xMaxCurr = (x == h-1 and xMin[windowIdx] <= 0 ? h+66 : xMax[windowIdx]);
                const int yMinCurr = (y == 0   and yMax[windowIdx] >= 0 ? 0    : yMin[windowIdx]);
                const int yMaxCurr = (y == w-1 and yMin[windowIdx] <= 0 ? w+66 : yMax[windowIdx]);

                t = max(0, min(x+xMinCurr, h) );
                b = max(0, min(x+xMaxCurr, h) );
                l = max(0, min(y+yMinCurr, w) );
                r = max(0, min(y+yMaxCurr, w) );

                gradInput[x*w + y] += 
                    ( gradOutputInt[b*(w+1) + r]
                    - gradOutputInt[t*(w+1) + r]
                    - gradOutputInt[b*(w+1) + l]
                    + gradOutputInt[t*(w+1) + l]);;
            }
        }

        // go to the next channel
        gradOutputInt += (h+1)*(w+1);
        gradOutput += h*gradOutputStride;
    }
}

void updateGradInputFrac(
    const float *gradOutputInt, const int nWindows, const int h, const int w, float *gradInput,
    const int *xMin, const int *xMax, const int *yMin, const int *yMax,
    const float *xMinFrac, const float *xMaxFrac, const float *yMinFrac, const float *yMaxFrac,
    const float *gradOutput, const int gradOutputStride,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        updateGradInputFracStrided(
            gradOutputInt, nWindows, h, w, gradInput,
            xMin, xMax, yMin, yMax, xMinFrac, xMaxFrac, yMinFrac, yMaxFrac,
            gradOutput, gradOutputStride, strideH, strideW);
        return;
    }
    
    int t, b, l, r;
    int tAdv, bAdv, lAdv, rAdv;
    int xMinCurr, xMaxCurr, yMinCurr, yMaxCurr;

    for (int windowIdx = 0; windowIdx < nWindows; ++windowIdx) {

        // #pragma omp parallel for private(t,b,l,r,xMinCurr,xMaxCurr,yMinCurr,yMaxCurr)
        for (int x = 0; x < h; ++x) {
            for (int y = 0; y < w; ++y) {

                xMinCurr = (x == 0   and xMax[windowIdx] >= 0 ? 0    : xMin[windowIdx]);
                xMaxCurr = (x == h-1 and xMin[windowIdx] <= 0 ? h+66 : xMax[windowIdx]);
                yMinCurr = (y == 0   and yMax[windowIdx] >= 0 ? 0    : yMin[windowIdx]);
                yMaxCurr = (y == w-1 and yMin[windowIdx] <= 0 ? w+66 : yMax[windowIdx]);

                t = max(0, min(x+xMinCurr, h) );
                b = max(0, min(x+xMaxCurr, h) );
                l = max(0, min(y+yMinCurr, w) );
                r = max(0, min(y+yMaxCurr, w) );

                tAdv = x+xMinCurr-1 <  h ? max(0, min(t-1, h)) : t;
                bAdv = x+xMaxCurr   >= 0 ? max(0, min(b+1, h)) : b;
                lAdv = y+yMinCurr-1 <  w ? max(0, min(l-1, w)) : l;
                rAdv = y+yMaxCurr   >= 0 ? max(0, min(r+1, w)) : r;

                gradInput[x*w + y] += 
                    ( gradOutputInt[b*(w+1) + r]
                    - gradOutputInt[t*(w+1) + r]
                    - gradOutputInt[b*(w+1) + l]
                    + gradOutputInt[t*(w+1) + l])
                   
                // -- xMax border
                +(gradOutputInt[bAdv*(w+1) + r]
                - gradOutputInt[b   *(w+1) + r]
                - gradOutputInt[bAdv*(w+1) + l]
                + gradOutputInt[b   *(w+1) + l]
                ) * xMaxFrac[windowIdx]

                // -- yMax border
                +(gradOutputInt[b*(w+1) + rAdv]
                - gradOutputInt[b*(w+1) + r   ]
                - gradOutputInt[t*(w+1) + rAdv]
                + gradOutputInt[t*(w+1) + r   ]
                ) * yMaxFrac[windowIdx]

                // -- xMin border
                +(gradOutputInt[t   *(w+1) + r]
                - gradOutputInt[tAdv*(w+1) + r]
                - gradOutputInt[t   *(w+1) + l]
                + gradOutputInt[tAdv*(w+1) + l]
                ) * xMinFrac[windowIdx]

                // -- yMin border
                +(gradOutputInt[b*(w+1) + l   ]
                - gradOutputInt[b*(w+1) + lAdv]
                - gradOutputInt[t*(w+1) + l   ]
                + gradOutputInt[t*(w+1) + lAdv]
                ) * yMinFrac[windowIdx]

                // -- corner pixels
                // Note we substitute using `bAdv-1 == b` and `rAdv-1 == r`
                + xMaxFrac[windowIdx]*yMaxFrac[windowIdx] * (
                       (x+xMaxCurr > h-1 or
                        y+yMaxCurr > w-1 or
                        x+xMaxCurr < 0   or
                        y+yMaxCurr < 0   or
                        b == bAdv or
                        r == rAdv) ? 0 :
                   gradOutput[b*gradOutputStride + r])

                + xMinFrac[windowIdx]*yMaxFrac[windowIdx] * (
                       (x+xMinCurr-1 > h-1 or
                        y+yMaxCurr   > w-1 or
                        x+xMinCurr-1 < 0   or
                        y+yMaxCurr   < 0   or
                        t == tAdv or
                        r == rAdv) ? 0 :
                   gradOutput[tAdv*gradOutputStride + r])

                + xMaxFrac[windowIdx]*yMinFrac[windowIdx] * (
                       (x+xMaxCurr   > h-1 or
                        y+yMinCurr-1 > w-1 or
                        x+xMaxCurr   < 0   or
                        y+yMinCurr-1 < 0   or
                        b == bAdv or
                        l == lAdv) ? 0 :
                   gradOutput[b*gradOutputStride + lAdv])

                + xMinFrac[windowIdx]*yMinFrac[windowIdx] * (
                       (x+xMinCurr-1 > h-1 or
                        y+yMinCurr-1 > w-1 or
                        x+xMinCurr-1 < 0   or
                        y+yMinCurr-1 < 0   or
                        t == tAdv or
                        l == lAdv) ? 0 :
                   gradOutput[tAdv*gradOutputStride + lAdv]);
            }
        }

        // go to the next channel
        gradOutputInt += (h+1)*(w+1);
        gradOutput += h*gradOutputStride;
    }
}

void backwardNoNorm(
    float *intData, float *gradOutData, float scale,
    int nWindows, int h, int w,
    float *gradXMin, float *gradXMax, float *gradYMin, float *gradYMax,
    int *xMinInt, int *xMaxInt, int *yMinInt, int *yMaxInt,
    const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        backwardNoNormStrided(
            intData, gradOutData, scale, nWindows, h, w, gradXMin, gradXMax, gradYMin, gradYMax,
            xMinInt, xMaxInt, yMinInt, yMaxInt, strideH, strideW);
        return;
    }

    for (int windowIdx = 0; windowIdx < nWindows; ++windowIdx) {
    
        double xMaxDelta = 0;
        double xMinDelta = 0;
        double yMaxDelta = 0;
        double yMinDelta = 0;
    
        // #pragma omp parallel for reduction(+:xMaxDelta,xMinDelta,yMaxDelta,yMinDelta)
        for (int x = 1; x <= h; ++x) {
            for (int y = 1; y <= w; ++y) {

                // `x+xMinInt[windowIdx]` = 0-index of the first "full" box cell
                // `x+xMaxInt[windowIdx]` = 0-index of the last "full" box cell + 1
                // <=> these are C-style segment indices

                // When the "full" cell (integral) part is empty,
                // there are two (fractional) cell rows that share the box.
                // `x+xMaxInt[windowIdx]` is the 0-index of the second one.
                
                if (x+xMaxInt[windowIdx] >= 1 and x+xMaxInt[windowIdx] < h) {
                    xMaxDelta += gradOutData[(x-1)*w + (y-1)] *
                        ( intData[max(1,min(x+xMaxInt[windowIdx]+1, h))*(w+1)
                            + max(0,min(y+yMaxInt[windowIdx], w))]
                        - intData[max(0,min(x+xMaxInt[windowIdx]  , h))*(w+1)
                            + max(0,min(y+yMaxInt[windowIdx], w))]
                        - intData[max(1,min(x+xMaxInt[windowIdx]+1, h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx], w))]
                        + intData[max(0,min(x+xMaxInt[windowIdx]  , h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx], w))]);
                }

                if (x+xMinInt[windowIdx] >= 1 and x+xMinInt[windowIdx] < h) {
                    xMinDelta -= gradOutData[(x-1)*w + (y-1)] *
                        ( intData[max(0,min(x+xMinInt[windowIdx]  , h  ))*(w+1)
                            + max(0,min(y+yMaxInt[windowIdx], w))]
                        - intData[max(0,min(x+xMinInt[windowIdx]-1, h-1))*(w+1)
                            + max(0,min(y+yMaxInt[windowIdx], w))]
                        - intData[max(0,min(x+xMinInt[windowIdx]  , h  ))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx], w))]
                        + intData[max(0,min(x+xMinInt[windowIdx]-1, h-1))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx], w))]);
                }

                if (y+yMaxInt[windowIdx] >= 1 and y+yMaxInt[windowIdx] < w) {
                    yMaxDelta += gradOutData[(x-1)*w + (y-1)] *
                        ( intData[max(0,min(x+xMaxInt[windowIdx], h))*(w+1)
                            + max(1,min(y+yMaxInt[windowIdx]+1, w))]
                        - intData[max(0,min(x+xMaxInt[windowIdx], h))*(w+1)
                            + max(0,min(y+yMaxInt[windowIdx]  , w))]
                        - intData[max(0,min(x+xMinInt[windowIdx], h))*(w+1)
                            + max(1,min(y+yMaxInt[windowIdx]+1, w))]
                        + intData[max(0,min(x+xMinInt[windowIdx], h))*(w+1)
                            + max(0,min(y+yMaxInt[windowIdx]  , w))]);
                }
                
                if (y+yMinInt[windowIdx] >= 1 and y+yMinInt[windowIdx] < w) {
                    yMinDelta -= gradOutData[(x-1)*w + (y-1)] *
                        ( intData[max(0,min(x+xMaxInt[windowIdx], h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx]  , w  ))]
                        - intData[max(0,min(x+xMaxInt[windowIdx], h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx]-1, w-1))]
                        - intData[max(0,min(x+xMinInt[windowIdx], h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx]  , w  ))]
                        + intData[max(0,min(x+xMinInt[windowIdx], h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx]-1, w-1))]);
                }
            }
        }

        gradXMin[windowIdx] += scale * xMinDelta;
        gradXMax[windowIdx] += scale * xMaxDelta;
        gradYMin[windowIdx] += scale * yMinDelta;
        gradYMax[windowIdx] += scale * yMaxDelta;

        gradOutData += h*w;
    }
}

// just for gradient debugging, add corners delta
void backwardNoNormFrac(
    float *intData, float *gradOutData, float scale,
    int nWindows, int h, int w,
    float *gradXMin, float *gradXMax, float *gradYMin, float *gradYMax,
    int *xMinInt, int *xMaxInt, int *yMinInt, int *yMaxInt,
    float *xMinFrac, float *xMaxFrac, float *yMinFrac, float *yMaxFrac,
    float *inData, int inStrideRow, const int strideH, const int strideW) {

    if (strideH != 1 or strideW != 1) {
        backwardNoNormFracStrided(
            intData, gradOutData, scale, nWindows, h, w, gradXMin, gradXMax, gradYMin, gradYMax,
            xMinInt, xMaxInt, yMinInt, yMaxInt, xMinFrac, xMaxFrac, yMinFrac, yMaxFrac,
            inData, inStrideRow, strideH, strideW);
        return;
    }

    float tlCorner, trCorner, blCorner, brCorner; // values from inData

    for (int windowIdx = 0; windowIdx < nWindows; ++windowIdx) {
    
        double xMaxDelta = 0;
        double xMinDelta = 0;
        double yMaxDelta = 0;
        double yMinDelta = 0;
    
        // #pragma omp parallel for reduction(+:xMaxDelta,xMinDelta,yMaxDelta,yMinDelta)
        for (int x = 1; x <= h; ++x) {
            for (int y = 1; y <= w; ++y) {

                // `x+xMinInt[windowIdx]` = 0-index of the first "full" box cell
                // `x+xMaxInt[windowIdx]` = 0-index of the last "full" box cell + 1
                // <=> these are C-style segment indices

                // When the "full" cell (integral) part is empty,
                // there are two (fractional) cell rows that share the box.
                // `x+xMaxInt[windowIdx]` is the 0-index of the second one.
                
                tlCorner = y+yMinInt[windowIdx] <  1 or x+xMinInt[windowIdx] <  1 ? 0 :
                                 inData[
                                    max(0,min(h-1,x+xMinInt[windowIdx]-1)) * inStrideRow +
                                    max(0,min(w-1,y+yMinInt[windowIdx]-1))];
                blCorner = y+yMinInt[windowIdx] <  1 or x+xMaxInt[windowIdx] >= h ? 0 :
                                 inData[
                                    max(0,min(h-1,x+xMaxInt[windowIdx]  )) * inStrideRow +
                                    max(0,min(w-1,y+yMinInt[windowIdx]-1))];
                trCorner = y+yMaxInt[windowIdx] >= w or x+xMinInt[windowIdx] <  1 ? 0 :
                                 inData[
                                    max(0,min(h-1,x+xMinInt[windowIdx]-1)) * inStrideRow +
                                    max(0,min(w-1,y+yMaxInt[windowIdx]  ))];
                brCorner = y+yMaxInt[windowIdx] >= w or x+xMaxInt[windowIdx] >= h ? 0 :
                                 inData[
                                    max(0,min(h-1,x+xMaxInt[windowIdx]  )) * inStrideRow +
                                    max(0,min(w-1,y+yMaxInt[windowIdx]  ))];

                if (x+xMaxInt[windowIdx] >= 1 and x+xMaxInt[windowIdx] < h) {
                    xMaxDelta += gradOutData[(x-1)*w + (y-1)] *
                        ( intData[max(0,min(x+xMaxInt[windowIdx]+1, h))*(w+1) 
                            + max(0,min(y+yMaxInt[windowIdx], w))]
                        - intData[max(0,min(x+xMaxInt[windowIdx]  , h))*(w+1) 
                            + max(0,min(y+yMaxInt[windowIdx], w))]
                        - intData[max(0,min(x+xMaxInt[windowIdx]+1, h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx], w))]
                        + intData[max(0,min(x+xMaxInt[windowIdx]  , h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx], w))]
                        + brCorner * (y+yMaxInt[windowIdx] <  1 ? 1.0f : yMaxFrac[windowIdx])
                        + blCorner * (y+yMinInt[windowIdx] >= w ? 1.0f : yMinFrac[windowIdx]));
                }

                if (x+xMinInt[windowIdx] >= 1 and x+xMinInt[windowIdx] < h) {
                    xMinDelta -= gradOutData[(x-1)*w + (y-1)] *
                        ( intData[max(0,min(x+xMinInt[windowIdx]  , h))*(w+1)
                            + max(0,min(y+yMaxInt[windowIdx], w))]
                        - intData[max(0,min(x+xMinInt[windowIdx]-1, h))*(w+1)
                            + max(0,min(y+yMaxInt[windowIdx], w))]
                        - intData[max(0,min(x+xMinInt[windowIdx]  , h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx], w))]
                        + intData[max(0,min(x+xMinInt[windowIdx]-1, h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx], w))]
                        + trCorner * (y+yMaxInt[windowIdx] <  1 ? 1.0f : yMaxFrac[windowIdx]) +
                        + tlCorner * (y+yMinInt[windowIdx] >= w ? 1.0f : yMinFrac[windowIdx]));
                }

                if (y+yMaxInt[windowIdx] >= 1 and y+yMaxInt[windowIdx] < w) {
                    yMaxDelta += gradOutData[(x-1)*w + (y-1)] *
                        ( intData[max(0,min(x+xMaxInt[windowIdx], h))*(w+1)
                            + max(0,min(y+yMaxInt[windowIdx]+1, w))]
                        - intData[max(0,min(x+xMaxInt[windowIdx], h))*(w+1)
                            + max(0,min(y+yMaxInt[windowIdx]  , w))]
                        - intData[max(0,min(x+xMinInt[windowIdx], h))*(w+1)
                            + max(0,min(y+yMaxInt[windowIdx]+1, w))]
                        + intData[max(0,min(x+xMinInt[windowIdx], h))*(w+1)
                            + max(0,min(y+yMaxInt[windowIdx]  , w))]
                        + trCorner * (x+xMinInt[windowIdx] >= h ? 1.0f : xMinFrac[windowIdx]) +
                        + brCorner * (x+xMaxInt[windowIdx] <  1 ? 1.0f : xMaxFrac[windowIdx]));
                }
                
                if (y+yMinInt[windowIdx] >= 1 and y+yMinInt[windowIdx] < w) {
                    yMinDelta -= gradOutData[(x-1)*w + (y-1)] *
                        ( intData[max(0,min(x+xMaxInt[windowIdx], h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx]  , w))]
                        - intData[max(0,min(x+xMaxInt[windowIdx], h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx]-1, w))]
                        - intData[max(0,min(x+xMinInt[windowIdx], h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx]  , w))]
                        + intData[max(0,min(x+xMinInt[windowIdx], h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx]-1, w))]
                        + tlCorner * (x+xMinInt[windowIdx] >= h ? 1.0f : xMinFrac[windowIdx]) +
                        + blCorner * (x+xMaxInt[windowIdx] <  1 ? 1.0f : xMaxFrac[windowIdx]));
                }
            }
        }

        gradXMin[windowIdx] += scale * xMinDelta;
        gradXMax[windowIdx] += scale * xMaxDelta;
        gradYMin[windowIdx] += scale * yMinDelta;
        gradYMax[windowIdx] += scale * yMaxDelta;

        gradOutData += h*w;
    }
}

}
