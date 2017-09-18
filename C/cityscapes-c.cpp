extern "C" {

void updateConfusionMatrix(
    long *confMatrix, const long *predictedLabels,
    const unsigned char *labels, const int numPixels) {
    
    long classPredicted, classTrue;
    for (int i = 0; i < numPixels; ++i) {
        classPredicted = predictedLabels[i]-1;
        classTrue = labels[i]-1;

        if (classTrue != 254) {
            ++confMatrix[classTrue*19 + classPredicted];
        }
    }
}

} // extern "C"