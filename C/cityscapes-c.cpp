extern "C" {

void updateConfusionMatrix(
    long *confMatrix, const long *predictedLabels,
    const long *labels, const int numPixels,
    const int nClasses) {
    
    long classPredicted, classTrue;
    for (int i = 0; i < numPixels; ++i) {
        classPredicted = predictedLabels[i]-1;
        classTrue = labels[i]-1;

        if (classTrue != nClasses and classPredicted != nClasses) {
            ++confMatrix[classTrue*nClasses + classPredicted];
        }
    }
}

} // extern "C"