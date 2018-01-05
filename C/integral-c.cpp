#include <algorithm>
#include <cmath>
#include <iostream>

using std::max;
using std::min;

int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

// Divides x by y (y > 0), rounds towards minus infinity 
inline
int divFloor(const int x, const int y) {
    return x >= 0 ? x / y : (x - y + 1) / y;
}

// Divides x by y (y > 0), rounds towards minus infinity, returns positive remainder
inline
int modFloor(const int x, const int y) {
    return x >= 0 ? x % y : (y + x % y);
}

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

void forwardNoNormReplicate(
    const float *intData, const int h, const int w, float *outData,
    const int xMinCurr, const int xMaxCurr, const int yMinCurr, const int yMaxCurr,
    const int strideH, const int strideW) {

    int t, b, l, r;
    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;

    // #pragma omp parallel for private(t,b,l,r)
    for (int x = 0; x < hOut; ++x) {
        for (int y = 0; y < wOut; ++y) {

            // note `1` / `h-1` / `w-1` because of "replicate" interpolation
            t = max(0, min(x*strideH+xMinCurr, h-1) );
            b = max(1, min(x*strideH+xMaxCurr, h)   );
            l = max(0, min(y*strideW+yMinCurr, w-1) );
            r = max(1, min(y*strideW+yMaxCurr, w)   );

            outData[x*wOut + y] = 
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
    const float *inData, const int inDataStride, const int strideH, const int strideW) {

    int t, b, l, r;
    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;

    // #pragma omp parallel for private(t,b,l,r)
    for (int x = 0; x < hOut; ++x) {
        for (int y = 0; y < wOut; ++y) {

            // note `1` / `h-1` / `w-1` because of "replicate" interpolation
            t = max(0, min(x*strideH+xMinCurr, h-1) );
            b = max(1, min(x*strideH+xMaxCurr, h)   );
            l = max(0, min(y*strideW+yMinCurr, w-1) );
            r = max(1, min(y*strideW+yMaxCurr, w)   );

            outData[x*wOut + y] = 
                ( intData[b*(w+1) + r]
                - intData[t*(w+1) + r]
                - intData[b*(w+1) + l]
                + intData[t*(w+1) + l])

            // -- xMax border
            +(intData[max(1,min(x*strideH+xMaxCurr+1,h))*(w+1) 
                + max(1,min(y*strideW+yMaxCurr,w))]
            - intData[max(1,min(x*strideH+xMaxCurr,h))*(w+1)
                + max(1,min(y*strideW+yMaxCurr,w))]
            - intData[max(1,min(x*strideH+xMaxCurr+1,h))*(w+1)
                + max(0,min(y*strideW+yMinCurr,w-1))]
            + intData[max(1,min(x*strideH+xMaxCurr,h))*(w+1)
                + max(0,min(y*strideW+yMinCurr,w-1))]
            ) * xMaxCurrFrac

            // -- yMax border
            +(intData[max(1,min(x*strideH+xMaxCurr,h))*(w+1) 
                + max(1,min(y*strideW+yMaxCurr+1,w))]
            - intData[max(1,min(x*strideH+xMaxCurr,h))*(w+1)
                + max(1,min(y*strideW+yMaxCurr,w))]
            - intData[max(0,min(x*strideH+xMinCurr,h-1))*(w+1)
                + max(1,min(y*strideW+yMaxCurr+1,w))]
            + intData[max(0,min(x*strideH+xMinCurr,h-1))*(w+1)
                + max(1,min(y*strideW+yMaxCurr,w))]
            ) * yMaxCurrFrac

            // -- xMin border
            +(intData[max(0,min(x*strideH+xMinCurr,h-1))*(w+1) 
                + max(1,min(y*strideW+yMaxCurr,w))]
            - intData[max(0,min(x*strideH+xMinCurr-1,h-1))*(w+1)
                + max(1,min(y*strideW+yMaxCurr,w))]
            - intData[max(0,min(x*strideH+xMinCurr,h-1))*(w+1)
                + max(0,min(y*strideW+yMinCurr,w-1))]
            + intData[max(0,min(x*strideH+xMinCurr-1,h-1))*(w+1)
                + max(0,min(y*strideW+yMinCurr,w-1))]
            ) * xMinCurrFrac

            // -- yMin border
            +(intData[max(1,min(x*strideH+xMaxCurr,h))*(w+1) 
                + max(0,min(y*strideW+yMinCurr,w-1))]
            - intData[max(1,min(x*strideH+xMaxCurr,h))*(w+1)
                + max(0,min(y*strideW+yMinCurr-1,w-1))]
            - intData[max(0,min(x*strideH+xMinCurr,h-1))*(w+1)
                + max(0,min(y*strideW+yMinCurr,w-1))]
            + intData[max(0,min(x*strideH+xMinCurr,h-1))*(w+1)
                + max(0,min(y*strideW+yMinCurr-1,w-1))]
            ) * yMinCurrFrac

            // -- corner pixels
            + xMaxCurrFrac*yMaxCurrFrac * (
                   (x*strideH+xMaxCurr >  h-1 or
                    y*strideW+yMaxCurr >  w-1 or
                    x*strideH+xMaxCurr <= 0   or
                    y*strideW+yMaxCurr <= 0) ? 0 : inData[(x*strideH+xMaxCurr)*inDataStride + (y*strideW+yMaxCurr)])

            + xMinCurrFrac*yMaxCurrFrac * (
                   (x*strideH+xMinCurr-1 >= h-1 or
                    y*strideW+yMaxCurr   >  w-1 or
                    x*strideH+xMinCurr-1 <  0   or
                    y*strideW+yMaxCurr   <= 0) ? 0 : inData[(x*strideH+xMinCurr-1)*inDataStride + (y*strideW+yMaxCurr)])

            + xMaxCurrFrac*yMinCurrFrac * (
                   (x*strideH+xMaxCurr   >  h-1 or
                    y*strideW+yMinCurr-1 >= w-1 or
                    x*strideH+xMaxCurr   <= 0   or
                    y*strideW+yMinCurr-1 <  0) ? 0 : inData[(x*strideH+xMaxCurr)*inDataStride + (y*strideW+yMinCurr-1)])

            + xMinCurrFrac*yMinCurrFrac * (
                   (x*strideH+xMinCurr-1 >= h-1 or
                    y*strideW+yMinCurr-1 >= w-1 or
                    x*strideH+xMinCurr-1 <  0   or
                    y*strideW+yMinCurr-1 <  0) ? 0 : inData[(x*strideH+xMinCurr-1)*inDataStride + (y*strideW+yMinCurr-1)]);

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
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float xMinCurrFrac, float xMaxCurrFrac, float yMinCurrFrac, float yMaxCurrFrac,
    float *inData, int inDataStride) {

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
    const float *gradOutputInt, const int nWindows, const int h, const int w, float *gradInput,
    const int *xMin, const int *xMax, const int *yMin, const int *yMax,
    const float *gradOutput, const int gradOutputStride, const int strideH, const int strideW) {

    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;
    int t, b, l, r;

    for (int windowIdx = 0; windowIdx < nWindows; ++windowIdx) {

        // #pragma omp parallel for private(t,b,l,r)
        for (int x = 0; x < h; ++x) {
            for (int y = 0; y < w; ++y) {

                // xMin, yMin: real inclusive lower border of the window
                // xMax, yMax: real inclusive upper border of the window plus 1
                const int xMinCurr = (x ==   0 and xMax[windowIdx] >= 0 ? 0    : xMin[windowIdx]);
                const int xMaxCurr = (x == h-1 and xMin[windowIdx] <= 0 ? h+66 : xMax[windowIdx]);
                const int yMinCurr = (y ==   0 and yMax[windowIdx] >= 0 ? 0    : yMin[windowIdx]);
                const int yMaxCurr = (y == w-1 and yMin[windowIdx] <= 0 ? w+66 : yMax[windowIdx]);

                t = max(0, min(divFloor(x+xMinCurr + strideH - 1, strideH)    , hOut) );
                b = max(0, min(divFloor(x+xMaxCurr - 1          , strideH) + 1, hOut) );
                l = max(0, min(divFloor(y+yMinCurr + strideW - 1, strideW)    , wOut) );
                r = max(0, min(divFloor(y+yMaxCurr - 1          , strideW) + 1, wOut) );

                gradInput[x*w + y] += 
                    ( gradOutputInt[b*(wOut+1) + r]
                    - gradOutputInt[t*(wOut+1) + r]
                    - gradOutputInt[b*(wOut+1) + l]
                    + gradOutputInt[t*(wOut+1) + l]);
            }
        }

        // go to the next channel
        gradOutputInt += (hOut+1)*(wOut+1);
    }
}

void updateGradInputFrac(
    const float *gradOutputInt, const int nWindows, const int h, const int w, float *gradInput,
    const int *xMin, const int *xMax, const int *yMin, const int *yMax,
    const float *xMinFrac, const float *xMaxFrac, const float *yMinFrac, const float *yMaxFrac,
    const float *gradOutput, const int gradOutputStride, const int strideH, const int strideW) {
    
    int t,    b,    l,    r   ; // border coordinates
    int tAdv, bAdv, lAdv, rAdv; // border coordinates advanced by 1
    int xMinCurr, xMaxCurr, yMinCurr, yMaxCurr;
    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;

    for (int windowIdx = 0; windowIdx < nWindows; ++windowIdx) {

        // #pragma omp parallel for private(t,b,l,r,xMinCurr,xMaxCurr,yMinCurr,yMaxCurr)
        for (int x = 0; x < h; ++x) {
            for (int y = 0; y < w; ++y) {

                // xMin, yMin: real inclusive lower border of the window
                // xMax, yMax: real inclusive upper border of the window plus 1
                const int xMinCurr = (x ==   0 and xMax[windowIdx] >= 0 ? 0    : xMin[windowIdx]);
                const int xMaxCurr = (x == h-1 and xMin[windowIdx] <= 0 ? h+66 : xMax[windowIdx]);
                const int yMinCurr = (y ==   0 and yMax[windowIdx] >= 0 ? 0    : yMin[windowIdx]);
                const int yMaxCurr = (y == w-1 and yMin[windowIdx] <= 0 ? w+66 : yMax[windowIdx]);

                // t,b,l,r: same but for gradOutput (which is reduced by stride)
                t = max(0, min(divFloor(x+xMinCurr + strideH - 1, strideH)    , hOut) );
                b = max(0, min(divFloor(x+xMaxCurr - 1          , strideH) + 1, hOut) );
                l = max(0, min(divFloor(y+yMinCurr + strideW - 1, strideW)    , wOut) );
                r = max(0, min(divFloor(y+yMaxCurr - 1          , strideW) + 1, wOut) );

                tAdv = modFloor(x+xMinCurr-1, strideH) == 0 and x+xMinCurr-1 <  h ? max(0, min(t-1, hOut)) : t;
                bAdv = modFloor(x+xMaxCurr  , strideH) == 0 and x+xMaxCurr   >= 0 ? max(0, min(b+1, hOut)) : b;
                lAdv = modFloor(y+yMinCurr-1, strideW) == 0 and y+yMinCurr-1 <  w ? max(0, min(l-1, wOut)) : l;
                rAdv = modFloor(y+yMaxCurr  , strideW) == 0 and y+yMaxCurr   >= 0 ? max(0, min(r+1, wOut)) : r;

                gradInput[x*w + y] += 
                    ( gradOutputInt[b*(wOut+1) + r]
                    - gradOutputInt[t*(wOut+1) + r]
                    - gradOutputInt[b*(wOut+1) + l]
                    + gradOutputInt[t*(wOut+1) + l])
                
                // -- xMax border
                +(gradOutputInt[bAdv*(wOut+1)
                    + r]
                - gradOutputInt[b*(wOut+1)
                    + r]
                - gradOutputInt[bAdv*(wOut+1)
                    + l]
                + gradOutputInt[b*(wOut+1)
                    + l]
                ) * xMaxFrac[windowIdx]

                // -- yMax border
                +(gradOutputInt[b*(wOut+1)
                    + rAdv]
                - gradOutputInt[b*(wOut+1)
                    + r]
                - gradOutputInt[t*(wOut+1)
                    + rAdv]
                + gradOutputInt[t*(wOut+1)
                    + r]
                ) * yMaxFrac[windowIdx]

                // -- xMin border
                +(gradOutputInt[t*(wOut+1)
                    + r]
                - gradOutputInt[tAdv*(wOut+1)
                    + r]
                - gradOutputInt[t*(wOut+1)
                    + l]
                + gradOutputInt[tAdv*(wOut+1)
                    + l]
                ) * xMinFrac[windowIdx]

                // -- yMin border
                +(gradOutputInt[b*(wOut+1)
                    + l]
                - gradOutputInt[b*(wOut+1)
                    + lAdv]
                - gradOutputInt[t*(wOut+1)
                    + l]
                + gradOutputInt[t*(wOut+1)
                    + lAdv]
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
        gradOutputInt += (hOut+1)*(wOut+1);
        gradOutput += hOut*gradOutputStride;
    }
}

void backwardNoNorm(
    float *intData, float *gradOutData, float scale,
    int nWindows, int h, int w,
    float *gradXMin, float *gradXMax, float *gradYMin, float *gradYMax,
    int *xMinInt, int *xMaxInt, int *yMinInt, int *yMaxInt,
    const int strideH, const int strideW) {

    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;

    for (int windowIdx = 0; windowIdx < nWindows; ++windowIdx) {
    
        double xMaxDelta = 0;
        double xMinDelta = 0;
        double yMaxDelta = 0;
        double yMinDelta = 0;
    
        // #pragma omp parallel for reduction(+:xMaxDelta,xMinDelta,yMaxDelta,yMinDelta)
        for (int xOut = 0; xOut < hOut; ++xOut) {
            for (int yOut = 0; yOut < wOut; ++yOut) {

                const int x = xOut*strideH + 1;
                const int y = yOut*strideW + 1;

                // `x+xMinInt[windowIdx]` = 0-index of the first "full" box cell
                // `x+xMaxInt[windowIdx]` = 0-index of the last "full" box cell + 1
                // <=> these are C-style segment indices

                // When the "full" cell (integral) part is empty,
                // there are two (fractional) cell rows that share the box.
                // `x+xMaxInt[windowIdx]` is the 0-index of the second one.
                
                if (x+xMaxInt[windowIdx] >= 1 and x+xMaxInt[windowIdx] < h) {
                    xMaxDelta += gradOutData[xOut*wOut + yOut] *
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
                    xMinDelta -= gradOutData[xOut*wOut + yOut] *
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
                    yMaxDelta += gradOutData[xOut*wOut + yOut] *
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
                    yMinDelta -= gradOutData[xOut*wOut + yOut] *
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

        gradOutData += hOut*wOut;
    }
}

// just for gradient debugging, add corners delta
void backwardNoNormFrac(
    float *intData, float *gradOutData, float scale,
    int nWindows, int h, int w,
    float *gradXMin, float *gradXMax, float *gradYMin, float *gradYMax,
    int *xMinInt, int *xMaxInt, int *yMinInt, int *yMaxInt,
    float *xMinFrac, float *xMaxFrac, float *yMinFrac, float *yMaxFrac,
    float *inData, int inStrideRow,
    const int strideH, const int strideW) {

    const int hOut = (h + strideH - 1) / strideH;
    const int wOut = (w + strideW - 1) / strideW;

    float tlCorner, trCorner, blCorner, brCorner; // values from inData

    for (int windowIdx = 0; windowIdx < nWindows; ++windowIdx) {
    
        double xMaxDelta = 0;
        double xMinDelta = 0;
        double yMaxDelta = 0;
        double yMinDelta = 0;
    
        // #pragma omp parallel for reduction(+:xMaxDelta,xMinDelta,yMaxDelta,yMinDelta)
        for (int xOut = 0; xOut < hOut; ++xOut) {
            for (int yOut = 0; yOut < wOut; ++yOut) {

                const int x = xOut*strideH + 1;
                const int y = yOut*strideW + 1;

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
                    xMaxDelta += gradOutData[xOut*wOut + yOut] *
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
                    xMinDelta -= gradOutData[xOut*wOut + yOut] *
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
                    yMaxDelta += gradOutData[xOut*wOut + yOut] *
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
                    yMinDelta -= gradOutData[xOut*wOut + yOut] *
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

        gradOutData += hOut*wOut;
    }
}

}
