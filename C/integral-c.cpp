#include <algorithm>
#include <iostream>
#include <omp.h>

using std::max;
using std::min;

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

int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
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

            int t = max(0, min(x+xMinCurr, h) );
            int b = max(0, min(x+xMaxCurr, h) );
            int l = max(0, min(y+yMinCurr, w) );
            int r = max(0, min(y+yMaxCurr, w) );
            // float coeff = gradOutData[(x-1)*w + (y-1)];

            xMaxDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[min(b+1, h)*(w+1) + r]
                - intData[b*(w+1) + r]
                - intData[min(b+1, h)*(w+1) + max(l-1, 0)]
                + intData[b*(w+1) + max(l-1, 0)] );
            
            xMinDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[max(t-1, 0)*(w+1) + r]
                - intData[t*(w+1) + r]
                - intData[max(t-1, 0)*(w+1) + max(l-1, 0)]
                + intData[t*(w+1) + max(l-1, 0)] );
            
            yMaxDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[b*(w+1) + min(r+1, w)]
                - intData[b*(w+1) + r]
                - intData[max(t-1, 0)*(w+1) + min(r+1, w)]
                + intData[max(t-1, 0)*(w+1) + r] );
            
            yMinDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[b*(w+1) + max(l-1, 0)]
                - intData[b*(w+1) + l]
                - intData[max(t-1, 0)*(w+1) + max(l-1, 0)]
                + intData[max(t-1, 0)*(w+1) + l] );
        }
    }

    deltas[1] = xMaxDelta;
    deltas[0] = xMinDelta;
    deltas[3] = yMaxDelta;
    deltas[2] = yMinDelta;
}

}