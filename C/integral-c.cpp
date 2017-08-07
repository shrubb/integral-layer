#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>

using std::max;
using std::min;

int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

extern "C" {

void forward(
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr, float areaCoeff) {

    int t, b, l, r;

    #pragma omp parallel for private(t,b,l,r)
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

    #pragma omp parallel for reduction(+:xMaxDelta,xMinDelta,yMaxDelta,yMinDelta)
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

void forwardNoNormReplicate(
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    int t, b, l, r;

    #pragma omp parallel for private(t,b,l,r)
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
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float xMinCurrFrac, float xMaxCurrFrac, float yMinCurrFrac, float yMaxCurrFrac,
    float *inData, int inDataStride) {

    int t, b, l, r;

    #pragma omp parallel for private(t,b,l,r)
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

            // if (x == 1 and y == 1) {
            //     std::cout << "outData[" << x << "," << y << "] = " << std::endl <<
            //     "+intData[" << b << "," << r << "] = " << intData[b*(w+1) + r] << std::endl <<
            //     "-intData[" << t << "," << r << "] = " << intData[t*(w+1) + r] << std::endl <<
            //     "-intData[" << b << "," << l << "] = " << intData[b*(w+1) + l] << std::endl <<
            //     "+intData[" << t << "," << l << "] = " << intData[t*(w+1) + l] << std::endl <<

            //     "+" << (intData[max(0,min(x+xMaxCurr+1,h))*(w+1) 
            //     + max(1,min(y+yMaxCurr,w))]
            // - intData[max(1,min(x+xMaxCurr,h))*(w+1)
            //     + max(1,min(y+yMaxCurr,w))]
            // - intData[max(1,min(x+xMaxCurr+1,h))*(w+1)
            //     + max(0,min(y+yMinCurr,w-1))]
            // + intData[max(1,min(x+xMaxCurr,h))*(w+1)
            //     + max(0,min(y+yMinCurr,w-1))]
            // ) << " * " << xMaxCurrFrac << std::endl <<

            //     "+" << (intData[max(1,min(x+xMaxCurr,h))*(w+1) 
            //     + max(1,min(y+yMaxCurr+1,w))]
            // - intData[max(1,min(x+xMaxCurr,h))*(w+1)
            //     + max(1,min(y+yMaxCurr,w))]
            // - intData[max(0,min(x+xMinCurr,h-1))*(w+1)
            //     + max(1,min(y+yMaxCurr+1,w))]
            // + intData[max(0,min(x+xMinCurr,h-1))*(w+1)
            //     + max(1,min(y+yMaxCurr,w))]
            // ) << " * " << yMaxCurrFrac << std::endl <<

            //     "+" << (intData[max(0,min(x+xMinCurr,h-1))*(w+1) 
            //     + max(1,min(y+yMaxCurr,w))]
            // - intData[max(0,min(x+xMinCurr-1,h-1))*(w+1)
            //     + max(1,min(y+yMaxCurr,w))]
            // - intData[max(0,min(x+xMinCurr,h-1))*(w+1)
            //     + max(0,min(y+yMinCurr,w-1))]
            // + intData[max(0,min(x+xMinCurr-1,h-1))*(w+1)
            //     + max(0,min(y+yMinCurr,w-1))]
            // ) << " * " << xMinCurrFrac << std::endl <<

            //     "+" << (intData[max(1,min(x+xMaxCurr,h))*(w+1) 
            //     + max(0,min(y+yMinCurr,w-1))]
            // - intData[max(1,min(x+xMaxCurr,h))*(w+1)
            //     + max(0,min(y+yMinCurr-1,w-1))]
            // - intData[max(0,min(x+xMinCurr,h-1))*(w+1)
            //     + max(0,min(y+yMinCurr,w-1))]
            // + intData[max(0,min(x+xMinCurr,h-1))*(w+1)
            //     + max(0,min(y+yMinCurr-1,w-1))]
            // ) << " * " << yMinCurrFrac << std::endl <<

            // // -- corner pixels
            // "+" << xMaxCurrFrac << " * " << yMaxCurrFrac << " * " << (
            //        (x+xMaxCurr >  h-1 or
            //         y+yMaxCurr >  w-1 or
            //         x+xMaxCurr <= 0   or
            //         y+yMaxCurr <= 0) ? 0 : inData[(x+xMaxCurr)*inDataStride + (y+yMaxCurr)]) << std::endl <<

            // "+" << xMinCurrFrac << " * " << yMaxCurrFrac << " * " << (
            //        (x+xMinCurr-1 >= h-1 or
            //         y+yMaxCurr   >  w-1 or
            //         x+xMinCurr-1 <  0   or
            //         y+yMaxCurr   <= 0) ? 0 : inData[(x+xMinCurr-1)*inDataStride + (y+yMaxCurr)]) << std::endl <<

            // "+" << xMaxCurrFrac << " * " << yMinCurrFrac << " * " << (
            //        (x+xMaxCurr   >  h-1 or
            //         y+yMinCurr-1 >= w-1 or
            //         x+xMaxCurr   <= 0   or
            //         y+yMinCurr-1 <  0) ? 0 : inData[(x+xMaxCurr)*inDataStride + (y+yMinCurr-1)]) << std::endl <<

            // "+" << xMinCurrFrac << " * " << yMinCurrFrac << " * " << (
            //        (x+xMinCurr-1 >= h-1 or
            //         y+yMinCurr-1 >= w-1 or
            //         x+xMinCurr-1 <  0   or
            //         y+yMinCurr-1 <  0) ? 0 : inData[(x+xMinCurr-1)*inDataStride + (y+yMinCurr-1)]) << std::endl <<

            //     "= " << outData[x*w + y] << std::endl;
            // }
        }
    }                            
}

void forwardNoNorm(
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    int t, b, l, r;

    #pragma omp parallel for private(t,b,l,r)
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
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float xMinCurrFrac, float xMaxCurrFrac, float yMinCurrFrac, float yMaxCurrFrac,
    float *inData, int inDataStride) {

    int t, b, l, r;

    #pragma omp parallel for private(t,b,l,r)
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
                + intData[t*(w+1) + l])

            // -- xMax border
            +(intData[max(0,min(x+xMaxCurr+1,h))*(w+1)
                + max(0,min(y+yMaxCurr,w))]
            - intData[max(0,min(x+xMaxCurr,h))*(w+1)
                + max(0,min(y+yMaxCurr,w))]
            - intData[max(0,min(x+xMaxCurr+1,h))*(w+1)
                + max(0,min(y+yMinCurr,w))]
            + intData[max(0,min(x+xMaxCurr,h))*(w+1)
                + max(0,min(y+yMinCurr,w))]
            ) * xMaxCurrFrac

            // -- yMax border
            +(intData[max(0,min(x+xMaxCurr,h))*(w+1)
                + max(0,min(y+yMaxCurr+1,w))]
            - intData[max(0,min(x+xMaxCurr,h))*(w+1)
                + max(0,min(y+yMaxCurr,w))]
            - intData[max(0,min(x+xMinCurr,h))*(w+1)
                + max(0,min(y+yMaxCurr+1,w))]
            + intData[max(0,min(x+xMinCurr,h))*(w+1)
                + max(0,min(y+yMaxCurr,w))]
            ) * yMaxCurrFrac

            // -- xMin border
            +(intData[max(0,min(x+xMinCurr,h))*(w+1)
                + max(0,min(y+yMaxCurr,w))]
            - intData[max(0,min(x+xMinCurr-1,h))*(w+1)
                + max(0,min(y+yMaxCurr,w))]
            - intData[max(0,min(x+xMinCurr,h))*(w+1)
                + max(0,min(y+yMinCurr,w))]
            + intData[max(0,min(x+xMinCurr-1,h))*(w+1)
                + max(0,min(y+yMinCurr,w))]
            ) * xMinCurrFrac

            // -- yMin border
            +(intData[max(0,min(x+xMaxCurr,h))*(w+1)
                + max(0,min(y+yMinCurr,w))]
            - intData[max(0,min(x+xMaxCurr,h))*(w+1)
                + max(0,min(y+yMinCurr-1,w))]
            - intData[max(0,min(x+xMinCurr,h))*(w+1)
                + max(0,min(y+yMinCurr,w))]
            + intData[max(0,min(x+xMinCurr,h))*(w+1)
                + max(0,min(y+yMinCurr-1,w))]
            ) * yMinCurrFrac

            // -- corner pixels
            + xMaxCurrFrac*yMaxCurrFrac * (
                   (x+xMaxCurr > h-1 or
                    y+yMaxCurr > w-1 or
                    x+xMaxCurr < 0   or
                    y+yMaxCurr < 0) ? 0 : inData[(x+xMaxCurr)*inDataStride + (y+yMaxCurr)])

            + xMinCurrFrac*yMaxCurrFrac * (
                   (x+xMinCurr-1 > h-1 or
                    y+yMaxCurr   > w-1 or
                    x+xMinCurr-1 < 0   or
                    y+yMaxCurr   < 0) ? 0 : inData[(x+xMinCurr-1)*inDataStride + (y+yMaxCurr)])

            + xMaxCurrFrac*yMinCurrFrac * (
                   (x+xMaxCurr   > h-1 or
                    y+yMinCurr-1 > w-1 or
                    x+xMaxCurr   < 0   or
                    y+yMinCurr-1 < 0) ? 0 : inData[(x+xMaxCurr)*inDataStride + (y+yMinCurr-1)])

            + xMinCurrFrac*yMinCurrFrac * (
                   (x+xMinCurr-1 > h-1 or
                    y+yMinCurr-1 > w-1 or
                    x+xMinCurr-1 < 0   or
                    y+yMinCurr-1 < 0) ? 0 : inData[(x+xMinCurr-1)*inDataStride + (y+yMinCurr-1)]);
        }
    }
}

void updateGradInput(
    float *gradOutputInt, int nWindows, int h, int w, float *gradInput,
    int *xMin, int *xMax, int *yMin, int *yMax,
    float *gradOutput, int gradOutputStride) {

    int t, b, l, r;

    for (int windowIdx = 0; windowIdx < nWindows; ++windowIdx) {

        #pragma omp parallel for private(t,b,l,r)
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

                // if (x == 1 and y == 1) {
                //     std::cout << "gradInput[x*w + y] += " << std::endl;
                //     std::cout << "+ gradOutputInt[b*(w+1) + r] = " << gradOutputInt[b*(w+1) + r] << std::endl;
                //     std::cout << "- gradOutputInt[t*(w+1) + r] = " << gradOutputInt[t*(w+1) + r] << std::endl;
                //     std::cout << "- gradOutputInt[b*(w+1) + l] = " << gradOutputInt[b*(w+1) + l] << std::endl;
                //     std::cout << "+ gradOutputInt[t*(w+1) + l] = " << gradOutputInt[t*(w+1) + l] << std::endl;

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
    float *gradOutputInt, int nWindows, int h, int w, float *gradInput,
    int *xMin, int *xMax, int *yMin, int *yMax,
    float *xMinFrac, float *xMaxFrac, float *yMinFrac, float *yMaxFrac,
    float *gradOutput, int gradOutputStride) {
    
    int t, b, l, r;
    int xMinCurr, xMaxCurr, yMinCurr, yMaxCurr;

    for (int windowIdx = 0; windowIdx < nWindows; ++windowIdx) {

        #pragma omp parallel for private(t,b,l,r,xMinCurr,xMaxCurr,yMinCurr,yMaxCurr)
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

                // if (x == 1 and y == 1) {
                //     std::cout << "gradInput[x*w + y] += " << std::endl;
                //     std::cout << "+ gradOutputInt[b*(w+1) + r] = " << gradOutputInt[b*(w+1) + r] << std::endl;
                //     std::cout << "- gradOutputInt[t*(w+1) + r] = " << gradOutputInt[t*(w+1) + r] << std::endl;
                //     std::cout << "- gradOutputInt[b*(w+1) + l] = " << gradOutputInt[b*(w+1) + l] << std::endl;
                //     std::cout << "+ gradOutputInt[t*(w+1) + l] = " << gradOutputInt[t*(w+1) + l] << std::endl;

                //     std::cout << "adding xMax border:" << std::endl;
                //     std::cout << "+gradOutputInt[" << max(0,min(x+xMaxCurr+1,h)) << "," << max(0,min(y+yMaxCurr,w)) << "] = " << gradOutputInt[max(0,min(x+xMaxCurr+1,h))*(w+1) + max(0,min(y+yMaxCurr,w))] << std::endl;
                //     std::cout << "-gradOutputInt[" << max(0,min(x+xMaxCurr,h)) << "," << max(0,min(y+yMaxCurr,w)) << "] = " << gradOutputInt[max(0,min(x+xMaxCurr,h))*(w+1) + max(0,min(y+yMaxCurr,w))] << std::endl;
                //     std::cout << "-gradOutputInt[" << max(0,min(x+xMaxCurr+1,h)) << "," << max(0,min(y+yMinCurr,w)) << "] = " << gradOutputInt[max(0,min(x+xMaxCurr+1,h))*(w+1) + max(0,min(y+yMinCurr,w))] << std::endl;
                //     std::cout << "+gradOutputInt[" << max(0,min(x+xMaxCurr,h)) << "," << max(0,min(y+yMinCurr,w)) << "] = " << gradOutputInt[max(0,min(x+xMaxCurr,h))*(w+1) + max(0,min(y+yMinCurr,w))] << std::endl;
                //     std::cout << "* xMaxFrac[windowIdx] = " << xMaxFrac[windowIdx] << std::endl;
                // }

                gradInput[x*w + y] += 
                    ( gradOutputInt[b*(w+1) + r]
                    - gradOutputInt[t*(w+1) + r]
                    - gradOutputInt[b*(w+1) + l]
                    + gradOutputInt[t*(w+1) + l])
                   
                // -- xMax border
                +(gradOutputInt[max(0,min(x+xMaxCurr+1,h))*(w+1)
                    + max(0,min(y+yMaxCurr,w))]
                - gradOutputInt[max(0,min(x+xMaxCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr,w))]
                - gradOutputInt[(max(0,min(x+xMaxCurr+1,h)))*(w+1)
                    + max(0,min(y+yMinCurr,w))]
                + gradOutputInt[max(0,min(x+xMaxCurr,h))*(w+1)
                    + max(0,min(y+yMinCurr,w))]
                ) * xMaxFrac[windowIdx]

                // -- yMax border
                +(gradOutputInt[max(0,min(x+xMaxCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr+1,w))]
                - gradOutputInt[max(0,min(x+xMaxCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr,w))]
                - gradOutputInt[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr+1,w))]
                + gradOutputInt[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr,w))]
                ) * yMaxFrac[windowIdx]

                // -- xMin border
                +(gradOutputInt[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr,w))]
                - gradOutputInt[max(0,min(x+xMinCurr-1,h))*(w+1)
                    + max(0,min(y+yMaxCurr,w))]
                - gradOutputInt[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMinCurr,w))]
                + gradOutputInt[max(0,min(x+xMinCurr-1,h))*(w+1)
                    + max(0,min(y+yMinCurr,w))]
                ) * xMinFrac[windowIdx]

                // -- yMin border
                +(gradOutputInt[max(0,min(x+xMaxCurr,h))*(w+1)
                    + max(0,min(y+yMinCurr,w))]
                - gradOutputInt[max(0,min(x+xMaxCurr,h))*(w+1)
                    + max(0,min(y+yMinCurr-1,w))]
                - gradOutputInt[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMinCurr,w))]
                + gradOutputInt[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMinCurr-1,w))]
                ) * yMinFrac[windowIdx]

                // -- corner pixels
                + xMaxFrac[windowIdx]*yMaxFrac[windowIdx] * (
                       (x+xMaxCurr > h-1 or
                        y+yMaxCurr > w-1 or
                        x+xMaxCurr < 0   or
                        y+yMaxCurr < 0) ? 0 : 
                   gradOutput[(x+xMaxCurr)*gradOutputStride + (y+yMaxCurr)])

                + xMinFrac[windowIdx]*yMaxFrac[windowIdx] * (
                       (x+xMinCurr-1 > h-1 or
                        y+yMaxCurr   > w-1 or
                        x+xMinCurr-1 < 0   or
                        y+yMaxCurr   < 0) ? 0 : 
                   gradOutput[(x+xMinCurr-1)*gradOutputStride + (y+yMaxCurr)])

                + xMaxFrac[windowIdx]*yMinFrac[windowIdx] * (
                       (x+xMaxCurr   > h-1 or
                        y+yMinCurr-1 > w-1 or
                        x+xMaxCurr   < 0   or
                        y+yMinCurr-1 < 0) ? 0 : 
                   gradOutput[(x+xMaxCurr)*gradOutputStride + (y+yMinCurr-1)])

                + xMinFrac[windowIdx]*yMinFrac[windowIdx] * (
                       (x+xMinCurr-1 > h-1 or
                        y+yMinCurr-1 > w-1 or
                        x+xMinCurr-1 < 0   or
                        y+yMinCurr-1 < 0) ? 0 : 
                   gradOutput[(x+xMinCurr-1)*gradOutputStride + (y+yMinCurr-1)]);
            }
        }

        // go to the next channel
        gradOutputInt += (h+1)*(w+1);
        gradOutput += h*gradOutputStride;
    }
}
/*
void backwardNoNorm(

    ) {
    

    float xMaxDelta = 0;
    float xMinDelta = 0;
    float yMaxDelta = 0;
    float yMinDelta = 0;

    #pragma omp parallel for reduction(+:xMaxDelta,xMinDelta,yMaxDelta,yMinDelta)
    for (int x = 1; x <= h; ++x) {
        for (int y = 1; y <= w; ++y) {

            xMaxDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[max(0,min(x+xMaxCurr  , h))*(w+1) 
                    + max(0,min(y+yMaxCurr-1, w))]
                - intData[max(0,min(x+xMaxCurr-1, h))*(w+1) 
                    + max(0,min(y+yMaxCurr-1, w))]
                - intData[max(0,min(x+xMaxCurr  , h))*(w+1)
                    + max(0,min(y+yMinCurr, w))]
                + intData[max(0,min(x+xMaxCurr-1, h))*(w+1)
                    + max(0,min(y+yMinCurr, w))]);
            
            xMinDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[max(0,min(x+xMinCurr-1, h))*(w+1) 
                    + max(0,min(y+yMaxCurr-1, w))]
                - intData[min(h,max(x+xMinCurr  , 0))*(w+1)
                    + max(0,min(y+yMaxCurr-1, w))]
                - intData[max(0,min(x+xMinCurr-1, h))*(w+1)
                    + max(0,min(y+yMinCurr  , w))]
                + intData[min(h,max(x+xMinCurr  , 0))*(w+1)
                    + max(0,min(y+yMinCurr  , w))]);
            
            yMaxDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[max(0,min(x+xMaxCurr-1, h))*(w+1) 
                    + max(0,min(y+yMaxCurr,w))]
                - intData[max(0,min(x+xMaxCurr-1, h))*(w+1)
                    + max(0,min(y+yMaxCurr-1, w))]
                - intData[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr,w))]
                + intData[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr-1, w))]);
            
            yMinDelta += gradOutData[(x-1)*w + (y-1)] *
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

    deltas[1] = xMaxDelta;
    deltas[0] = xMinDelta;
    deltas[3] = yMaxDelta;
    deltas[2] = yMinDelta;
}*/

// just for gradient debugging, add corners delta
void backwardNoNormFrac(
    float *intData, float *gradOutData, float scale,
    int nWindows, int h, int w,
    float *gradXMin, float *gradXMax, float *gradYMin, float *gradYMax,
    int *xMinInt, int *xMaxInt, int *yMinInt, int *yMaxInt,
    float *xMinFrac, float *xMaxFrac, float *yMinFrac, float *yMaxFrac,
    float *inData, int inStrideChannel, int inStrideRow) {

    for (int windowIdx = 0; windowIdx < nWindows; ++windowIdx) {
    
        float xMaxDelta = 0;
        float xMinDelta = 0;
        float yMaxDelta = 0;
        float yMinDelta = 0;
    
        #pragma omp parallel for reduction(+:xMaxDelta,xMinDelta,yMaxDelta,yMinDelta)
        for (int x = 1; x <= h; ++x) {
            for (int y = 1; y <= w; ++y) {
                
                float tlCorner = (x+xMinInt[windowIdx] > h or y+yMinInt[windowIdx] > w or 
                                  x+xMinInt[windowIdx] < 1 or y+yMinInt[windowIdx] < 1) ? 0 :
                                  inData[(x+xMinInt[windowIdx]-1)*inStrideRow + (y+yMinInt[windowIdx]-1)];
                float blCorner = (x+xMaxInt[windowIdx] > h or y+yMinInt[windowIdx] > w or
                                  x+xMaxInt[windowIdx] < 1 or y+yMinInt[windowIdx] < 1) ? 0 :
                                  inData[(x+xMaxInt[windowIdx]-1)*inStrideRow + (y+yMinInt[windowIdx]-1)];
                float trCorner = (x+xMinInt[windowIdx] > h or y+yMaxInt[windowIdx] > w or
                                  x+xMinInt[windowIdx] < 1 or y+yMaxInt[windowIdx] < 1) ? 0 :
                                  inData[(x+xMinInt[windowIdx]-1)*inStrideRow + (y+yMaxInt[windowIdx]-1)];
                float brCorner = (x+xMaxInt[windowIdx] > h or y+yMaxInt[windowIdx] > w or
                                  x+xMaxInt[windowIdx] < 1 or y+yMaxInt[windowIdx] < 1) ? 0 :
                                  inData[(x+xMaxInt[windowIdx]-1)*inStrideRow + (y+yMaxInt[windowIdx]-1)];

                if (x+xMaxInt[windowIdx] > 1) {
                    xMaxDelta += gradOutData[(x-1)*w + (y-1)] *
                        ( intData[max(0,min(x+xMaxInt[windowIdx]  , h))*(w+1) 
                            + max(0,min(y+yMaxInt[windowIdx]-1, w))]
                        - intData[max(0,min(x+xMaxInt[windowIdx]-1, h))*(w+1) 
                            + max(0,min(y+yMaxInt[windowIdx]-1, w))]
                        - intData[max(0,min(x+xMaxInt[windowIdx]  , h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx], w))]
                        + intData[max(0,min(x+xMaxInt[windowIdx]-1, h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx], w))]
                        + brCorner * (y+yMaxInt[windowIdx] == 0 ? 1.0f : yMaxFrac[windowIdx])
                        + blCorner * (y+yMinInt[windowIdx] == w ? 1.0f : yMinFrac[windowIdx]));
                }

                if (x+xMinInt[windowIdx] < h) {
                    xMinDelta += gradOutData[(x-1)*w + (y-1)] *
                        ( intData[max(0,min(x+xMinInt[windowIdx]-1, h))*(w+1) 
                            + max(0,min(y+yMaxInt[windowIdx]-1, w))]
                        - intData[min(h,max(x+xMinInt[windowIdx]  , 0))*(w+1)
                            + max(0,min(y+yMaxInt[windowIdx]-1, w))]
                        - intData[max(0,min(x+xMinInt[windowIdx]-1, h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx]  , w))]
                        + intData[min(h,max(x+xMinInt[windowIdx]  , 0))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx]  , w))]
                        - trCorner * (y+yMaxInt[windowIdx] == 0 ? 1.0f : yMaxFrac[windowIdx])
                        - tlCorner * (y+yMinInt[windowIdx] == w ? 1.0f : yMinFrac[windowIdx]));
                }

                if (y+yMaxInt[windowIdx] > 1) {
                    yMaxDelta += gradOutData[(x-1)*w + (y-1)] *
                        ( intData[max(0,min(x+xMaxInt[windowIdx]-1, h))*(w+1) 
                            + max(0,min(y+yMaxInt[windowIdx],w))]
                        - intData[max(0,min(x+xMaxInt[windowIdx]-1, h))*(w+1)
                            + max(0,min(y+yMaxInt[windowIdx]-1, w))]
                        - intData[max(0,min(x+xMinInt[windowIdx],h))*(w+1)
                            + max(0,min(y+yMaxInt[windowIdx],w))]
                        + intData[max(0,min(x+xMinInt[windowIdx],h))*(w+1)
                            + max(0,min(y+yMaxInt[windowIdx]-1, w))]
                        + trCorner * (x+xMinInt[windowIdx] == h ? 1.0f : xMinFrac[windowIdx])
                        + brCorner * (x+xMaxInt[windowIdx] == 0 ? 1.0f : xMaxFrac[windowIdx]));
                }
                
                if (y+yMinInt[windowIdx] < w) {
                    yMinDelta += gradOutData[(x-1)*w + (y-1)] *
                        ( intData[max(0,min(x+xMaxInt[windowIdx]-1, h))*(w+1) 
                            + max(0,min(y+yMinInt[windowIdx]-1,w))]
                        - intData[max(0,min(x+xMaxInt[windowIdx]-1, h))*(w+1)
                            + min(w,max(y+yMinInt[windowIdx], 0))]
                        - intData[max(0,min(x+xMinInt[windowIdx]  , h))*(w+1)
                            + max(0,min(y+yMinInt[windowIdx]-1,w))]
                        + intData[max(0,min(x+xMinInt[windowIdx]  , h))*(w+1)
                            + min(w,max(y+yMinInt[windowIdx], 0))]
                        - tlCorner * (x+xMinInt[windowIdx] == h ? 1.0f : xMinFrac[windowIdx])
                        - blCorner * (x+xMaxInt[windowIdx] == 0 ? 1.0f : xMaxFrac[windowIdx]));
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
