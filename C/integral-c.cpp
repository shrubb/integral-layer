#include <algorithm>

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

void backward(
    float *intData, float *gradOutData, int h, int w, float *deltas,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr) {

    float *xMaxDelta = &deltas[1];
    float *xMinDelta = &deltas[0];
    float *yMaxDelta = &deltas[3];
    float *yMinDelta = &deltas[2];

    // #pragma omp parallel for reduction(+:...)
    for (int x = 1; x <= h; ++x) {
        for (int y = 1; y <= w; ++y) {

            int t = max(0, min(x+xMinCurr, h) );
            int b = max(0, min(x+xMaxCurr, h) );
            int l = max(0, min(y+yMinCurr, w) );
            int r = max(0, min(y+yMaxCurr, w) );

            *xMaxDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[min(b+1, h)*(w+1) + r]
                - intData[b*(w+1) + r]
                - intData[min(b+1, h)*(w+1) + max(l-1, 0)]
                + intData[b*(w+1) + max(l-1, 0)] );
            
            *xMinDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[max(t-1, 0)*(w+1) + r]
                - intData[t*(w+1) + r]
                - intData[max(t-1, 0)*(w+1) + max(l-1, 0)]
                + intData[t*(w+1) + max(l-1, 0)] );
            
            *yMaxDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[b*(w+1) + min(r+1, w)]
                - intData[b*(w+1) + r]
                - intData[max(t-1, 0)*(w+1) + min(r+1, w)]
                + intData[max(t-1, 0)*(w+1) + r] );
            
            *yMinDelta += gradOutData[(x-1)*w + (y-1)] *
                ( intData[b*(w+1) + max(l-1, 0)]
                - intData[b*(w+1) + l]
                - intData[max(t-1, 0)*(w+1) + max(l-1, 0)]
                + intData[max(t-1, 0)*(w+1) + l] );
        }
    }
}

}