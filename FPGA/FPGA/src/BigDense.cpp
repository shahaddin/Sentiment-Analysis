#include "BigDense.h"

void BigDense::forwardRelu(float* input, float* outputArrGiven){
    
    for(int i = 0; i< outputSize; i++){
        outputArrGiven[i] = dotCommon(weights[i], input, inputSize) + bias[i];

        //manual max() func implementing ReLU
        if(outputArrGiven[i] < 0){
            outputArrGiven[i] = 0;
            
        }
    }
    return;
};
