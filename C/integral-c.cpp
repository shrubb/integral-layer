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
    float *gradOutputInt, int channels, int h, int w,float *gradInput,
    int *xMin, int *xMax, int *yMin, int *yMax,
    float *gradOutput, int gradOutputStride) {

    int t, b, l, r;

    for (int ch = 0; ch < channels; ++ch) {

        #pragma omp parallel for private(t,b,l,r)
        for (int x = 0; x < h; ++x) {
            for (int y = 0; y < w; ++y) {

                const int xMinCurr = (x == 0   and xMax[ch] >= 0 ? 0    : xMin[ch]);
                const int xMaxCurr = (x == h-1 and xMin[ch] <= 0 ? h+66 : xMax[ch]);
                const int yMinCurr = (y == 0   and yMax[ch] >= 0 ? 0    : yMin[ch]);
                const int yMaxCurr = (y == w-1 and yMin[ch] <= 0 ? w+66 : yMax[ch]);

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
    float *gradOutputInt, int channels, int h, int w, float *gradInput,
    int *xMin, int *xMax, int *yMin, int *yMax,
    float *xMinFrac, float *xMaxFrac, float *yMinFrac, float *yMaxFrac,
    float *gradOutput, int gradOutputStride) {
    
    int t, b, l, r;
    int xMinCurr, xMaxCurr, yMinCurr, yMaxCurr;

    for (int ch = 0; ch < channels; ++ch) {

        #pragma omp parallel for private(t,b,l,r,xMinCurr,xMaxCurr,yMinCurr,yMaxCurr)
        for (int x = 0; x < h; ++x) {
            for (int y = 0; y < w; ++y) {

                xMinCurr = (x == 0   and xMax[ch] >= 0 ? 0    : xMin[ch]);
                xMaxCurr = (x == h-1 and xMin[ch] <= 0 ? h+66 : xMax[ch]);
                yMinCurr = (y == 0   and yMax[ch] >= 0 ? 0    : yMin[ch]);
                yMaxCurr = (y == w-1 and yMin[ch] <= 0 ? w+66 : yMax[ch]);

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
                //     std::cout << "* xMaxFrac[ch] = " << xMaxFrac[ch] << std::endl;
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
                ) * xMaxFrac[ch]

                // -- yMax border
                +(gradOutputInt[max(0,min(x+xMaxCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr+1,w))]
                - gradOutputInt[max(0,min(x+xMaxCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr,w))]
                - gradOutputInt[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr+1,w))]
                + gradOutputInt[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr,w))]
                ) * yMaxFrac[ch]

                // -- xMin border
                +(gradOutputInt[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMaxCurr,w))]
                - gradOutputInt[max(0,min(x+xMinCurr-1,h))*(w+1)
                    + max(0,min(y+yMaxCurr,w))]
                - gradOutputInt[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMinCurr,w))]
                + gradOutputInt[max(0,min(x+xMinCurr-1,h))*(w+1)
                    + max(0,min(y+yMinCurr,w))]
                ) * xMinFrac[ch]

                // -- yMin border
                +(gradOutputInt[max(0,min(x+xMaxCurr,h))*(w+1)
                    + max(0,min(y+yMinCurr,w))]
                - gradOutputInt[max(0,min(x+xMaxCurr,h))*(w+1)
                    + max(0,min(y+yMinCurr-1,w))]
                - gradOutputInt[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMinCurr,w))]
                + gradOutputInt[max(0,min(x+xMinCurr,h))*(w+1)
                    + max(0,min(y+yMinCurr-1,w))]
                ) * yMinFrac[ch]

                // -- corner pixels
                + xMaxFrac[ch]*yMaxFrac[ch] * (
                       (x+xMaxCurr > h-1 or
                        y+yMaxCurr > w-1 or
                        x+xMaxCurr < 0   or
                        y+yMaxCurr < 0) ? 0 : 
                   gradOutput[(x+xMaxCurr)*gradOutputStride + (y+yMaxCurr)])

                + xMinFrac[ch]*yMaxFrac[ch] * (
                       (x+xMinCurr-1 > h-1 or
                        y+yMaxCurr   > w-1 or
                        x+xMinCurr-1 < 0   or
                        y+yMaxCurr   < 0) ? 0 : 
                   gradOutput[(x+xMinCurr-1)*gradOutputStride + (y+yMaxCurr)])

                + xMaxFrac[ch]*yMinFrac[ch] * (
                       (x+xMaxCurr   > h-1 or
                        y+yMinCurr-1 > w-1 or
                        x+xMaxCurr   < 0   or
                        y+yMinCurr-1 < 0) ? 0 : 
                   gradOutput[(x+xMaxCurr)*gradOutputStride + (y+yMinCurr-1)])

                + xMinFrac[ch]*yMinFrac[ch] * (
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

void backwardNoNorm(
    float *intData, float *gradOutData, int h, int w, float *deltas,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

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
}

// just for gradient debugging, add corners delta
void backwardNoNormFrac(
    float *intData, float *gradOutData, int h, int w, float *deltas,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float xMinCurrFrac, float xMaxCurrFrac, float yMinCurrFrac, float yMaxCurrFrac,
    float *inData, int inDataStride) {

    float xMaxDelta = 0;
    float xMinDelta = 0;
    float yMaxDelta = 0;
    float yMinDelta = 0;

    #pragma omp parallel for reduction(+:xMaxDelta,xMinDelta,yMaxDelta,yMinDelta)
    for (int x = 1; x <= h; ++x) {
        for (int y = 1; y <= w; ++y) {
            
            float tlCorner = (x+xMinCurr > h or y+yMinCurr > w or 
                              x+xMinCurr < 1 or y+yMinCurr < 1) ? 0 :
                              inData[(x+xMinCurr-1)*inDataStride + (y+yMinCurr-1)];
            float blCorner = (x+xMaxCurr > h or y+yMinCurr > w or
                              x+xMaxCurr < 1 or y+yMinCurr < 1) ? 0 :
                              inData[(x+xMaxCurr-1)*inDataStride + (y+yMinCurr-1)];
            float trCorner = (x+xMinCurr > h or y+yMaxCurr > w or
                              x+xMinCurr < 1 or y+yMaxCurr < 1) ? 0 :
                              inData[(x+xMinCurr-1)*inDataStride + (y+yMaxCurr-1)];
            float brCorner = (x+xMaxCurr > h or y+yMaxCurr > w or
                              x+xMaxCurr < 1 or y+yMaxCurr < 1) ? 0 :
                              inData[(x+xMaxCurr-1)*inDataStride + (y+yMaxCurr-1)];

            xMaxDelta += gradOutData[(x-1)*w + (y-1)] *
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
            
            xMinDelta += gradOutData[(x-1)*w + (y-1)] *
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
            
            yMaxDelta += gradOutData[(x-1)*w + (y-1)] *
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
            
            yMinDelta += gradOutData[(x-1)*w + (y-1)] *
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

    deltas[1] = xMaxDelta;
    deltas[0] = xMinDelta;
    deltas[3] = yMaxDelta;
    deltas[2] = yMinDelta;
}

}
