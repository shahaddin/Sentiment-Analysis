#include "SmallDense.h"
#include <math.h>

SmallDense::SmallDense(int inpSize, int outSize) : inputSize(inpSize), outputSize(outSize) 
{
};


void SmallDense::forwardSoftmax(float* input, float* outputGiven){

    float totalExp{};
    float maxVal{};
    float intermediateVal{};
    for(int i = 0; i< outputSize; i++){
        intermediateVal = dotCommon(weights[i], input, inputSize) + bias[i];
        output[i] = intermediateVal;
        if (intermediateVal > maxVal){ maxVal = intermediateVal; }
    }

    // Subtract the maximum value for numerical stability
    for(int i = 0; i < outputSize; i++) {
        output[i] -= maxVal;
    }

    for(int i = 0; i < outputSize; i++) {
        output[i] = exp(output[i]);
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


