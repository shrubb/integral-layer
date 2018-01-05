extern "C" {

void forwardNoNormReplicateStrided(
    const float *intData, const int h, const int w, float *outData,
    const int xMinCurr, const int xMaxCurr, const int yMinCurr, const int yMaxCurr,
    const int strideH, const int strideW);

void forwardNoNormReplicateFracStrided(
    const float *intData, const int h, const int w, float *outData,
    const int xMinCurr, const int xMaxCurr, const int yMinCurr, const int yMaxCurr,
    const float xMinCurrFrac, const float xMaxCurrFrac,
    const float yMinCurrFrac, const float yMaxCurrFrac,
    const float *inData, const int inDataStride, const int strideH, const int strideW);

void updateGradInputStrided(
    const float *gradOutputInt, const int nWindows, const int h, const int w, float *gradInput,
    const int *xMin, const int *xMax, const int *yMin, const int *yMax,
    const float *gradOutput, const int gradOutputStride, const int strideH, const int strideW);

void updateGradInputFracStrided(
    const float *gradOutputInt, const int nWindows, const int h, const int w, float *gradInput,
    const int *xMin, const int *xMax, const int *yMin, const int *yMax,
    const float *xMinFrac, const float *xMaxFrac, const float *yMinFrac, const float *yMaxFrac,
    const float *gradOutput, const int gradOutputStride, const int strideH, const int strideW);

void backwardNoNormStrided(
    float *intData, float *gradOutData, float scale,
    int nWindows, int h, int w,
    float *gradXMin, float *gradXMax, float *gradYMin, float *gradYMax,
    int *xMinInt, int *xMaxInt, int *yMinInt, int *yMaxInt,
    const int strideH, const int strideW);

// just for gradient debugging, add corners delta
void backwardNoNormFracStrided(
    float *intData, float *gradOutData, float scale,
    int nWindows, int h, int w,
    float *gradXMin, float *gradXMax, float *gradYMin, float *gradYMax,
    int *xMinInt, int *xMaxInt, int *yMinInt, int *yMaxInt,
    float *xMinFrac, float *xMaxFrac, float *yMinFrac, float *yMaxFrac,
    float *inData, int inStrideRow,
    const int strideH, const int strideW);

}
