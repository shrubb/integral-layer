#include <algorithm>
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

#if 0
    std::cout << "gold tmpArray:" << std::endl;
    for (int x = 1; x <= h; ++x) {
        for (int y = 1; y <= w; ++y) {
            
            int tClip = max(x+xMinCurr, 0);
            int bClip = min(x+xMaxCurr, h);
            int lClip = max(y+yMinCurr, 0);
            int rClip = min(y+yMaxCurr, w);

            if (x >= 33 or y >= 33)
            std::cout <<    
        gradOutData[(x-1)*w + (y-1)] * ( intData[max(0,min(x+xMaxCurr+1,h))*(w+1) 
            + max(0,rClip)]
        - intData[max(0,bClip)*(w+1) 
            + max(0,rClip)]
        - intData[max(0,min(x+xMaxCurr+1,h))*(w+1)
            + max(0,min(y+yMinCurr-1,w))]
        + intData[max(0,bClip)*(w+1)
            + max(0,min(y+yMinCurr-1,w))] ) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
#endif

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

void forwardNoNorm(
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    int t, b, l, r;

    #pragma omp parallel for private(t,b,l,r)
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
                + intData[t*(w+1) + l]);
        }
    }
}

// just for gradient debugging
void forwardNoNormFrac(
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr,
    float xMinCurrFrac, float xMaxCurrFrac, float yMinCurrFrac, float yMaxCurrFrac,
    float *inData) {

    int t, b, l, r;

    #pragma omp parallel for private(t,b,l,r)
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
                   (x+xMaxCurr > w-1 or
                    y+yMaxCurr > h-1 or
                    x+xMaxCurr < 0   or
                    y+yMaxCurr < 0) ? 0 : inData[(x+xMaxCurr)*w + (y+yMaxCurr)])

            + xMinCurrFrac*yMaxCurrFrac * (
                   (x+xMinCurr-1 > w-1 or
                    y+yMaxCurr   > h-1 or
                    x+xMinCurr-1 < 0   or
                    y+yMaxCurr   < 0) ? 0 : inData[(x+xMinCurr-1)*w + (y+yMaxCurr)])

            + xMaxCurrFrac*yMinCurrFrac * (
                   (x+xMaxCurr   > w-1 or
                    y+yMinCurr-1 > h-1 or
                    x+xMaxCurr   < 0   or
                    y+yMinCurr-1 < 0) ? 0 : inData[(x+xMaxCurr)*w + (y+yMinCurr-1)])

            + xMinCurrFrac*yMinCurrFrac * (
                   (x+xMinCurr-1 > w-1 or
                    y+yMinCurr-1 > h-1 or
                    x+xMinCurr-1 < 0   or
                    y+yMinCurr-1 < 0) ? 0 : inData[(x+xMinCurr-1)*w + (y+yMinCurr-1)]);
        }
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

}
