#include "BigDense.h"
#include <math.h>

BigDense::BigDense(int inpSize, int outSize) : inputSize(inpSize), outputSize(outSize) 
{
};

void BigDense::forwardRelu(float* input, float* outputArrGiven){
    for(int i = 0; i< outputSize; i++){
        output[i] = max(0.0f, dotCommon(weights[i], input, inputSize) + bias[i]);
    }

    for(int i =0; i < outputSize; i++){
        outputArrGiven[i] = output[i];
    }

    return;
};
