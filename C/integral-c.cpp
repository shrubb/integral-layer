#include <algorithm>

using std::max;
using std::min;

extern "C" {

void forward(
    float *intData, int h, int w, float *outData,
    int xMinCurr, int xMaxCurr, int yMinCurr, int yMaxCurr, 
    float areaCoeff) {

    int t, b, l, r;

    #pragma omp parallel for private(t,b,l,r)
    for (int x = 0; x < h; ++x) {
        for (int y = 0; y < w; ++y) {

            t = max(0, min(x+xMinCurr, h) );
            b = max(0, min(x+xMaxCurr, h) );
            l = max(0, min(y+yMinCurr, w) );
            r = max(0, min(y+yMaxCurr, w) );

            outData[x*w + y] = areaCoeff * (
                  intData[b*(w+1) + r]
                - intData[t*(w+1) + r]
                - intData[b*(w+1) + l]
                + intData[t*(w+1) + l]);
        }
    }
}

}