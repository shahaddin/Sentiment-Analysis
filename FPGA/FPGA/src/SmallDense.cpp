#include "SmallDense.h"
#include <math.h>

void SmallDense::forwardSoftmax(float* input, float* outputGiven){

    float totalExp{};
    for(int i = 0; i< outputSize; i++){
        output[i] = exp(dotCommon(weights[i], input, inputSize) + bias[i]);
        totalExp += output[i];
    }

    for(int i = 0; i < outputSize; i++) {
        output[i] /= totalExp;
    }

    for(int i=0; i< outputSize; i++){
        outputGiven[i] = output[i];
    }

    return;
};


