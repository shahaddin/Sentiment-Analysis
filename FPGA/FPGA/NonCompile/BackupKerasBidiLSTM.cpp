#include "KerasBidiLSTM.h"


//returns a singular 128 length float vector
void LSTMCellForward::forward( float (*VecEmbdsInput)[LSTM_INPUT_SIZE]){
    // Implement forward pass using LSTM equations
    int iterationNum = WORD_NUMBER; // NUMBER OF WORDS LIMIT BEING PASSED
    for(int h = 0; h < iterationNum; h++){
        // so for these following dots Im gonna need to operate for specific amount starting from specific index of a row so

        float candidateCell[HIDDEN_SIZE];
        int hiddensizeTimes2 = hiddenSize*2;
        int hiddensizeTimes3 = hiddenSize*3;

        for(int i = 0; i < hiddenSize; i++)
        {
            inputGate[i] = sigmoid( dotCommon(weightsW[i], VecEmbdsInput[h], hiddenSize) + dotCommon(weightsU[i], hiddenVector, hiddenSize) + bias[i]);
            forgetGate[i] = sigmoid( dotCommon(weightsW[i]+hiddenSize, VecEmbdsInput[h], hiddenSize) + dotCommon(weightsU[i]+hiddenSize, hiddenVector, hiddenSize) + bias[hiddenSize + i]);
            candidateCell[i] = tanh( dotCommon(weightsW[i]+hiddensizeTimes2, VecEmbdsInput[h], hiddenSize) + dotCommon(weightsU[i]+hiddensizeTimes2, hiddenVector, hiddenSize) + bias[hiddensizeTimes2 + i]);
            outputGate[i] = sigmoid( dotCommon(weightsW[i]+hiddensizeTimes3, VecEmbdsInput[h], hiddenSize) + dotCommon(weightsU[i]+hiddensizeTimes3, hiddenVector, hiddenSize) + bias[hiddensizeTimes3 + i]);
        
            cellActivation[i] = (forgetGate[i] * cellActivation[i]) + (inputGate[i] * candidateCell[i]);

            hiddenVector[i] = outputGate[i] * tanh(cellActivation[i]);
        }
    }
    return;
};

//returns a singular 128 length float vector
void LSTMCellBackward::forward( float (*VecEmbdsInput)[LSTM_INPUT_SIZE]){
    // Implement forward pass using LSTM equations
    int iterationNum = WORD_NUMBER; // NUMBER OF WORDS LIMIT BEING PASSED
    for(int h = 0; h < iterationNum; h++){
        // so for these following dots Im gonna need to operate for specific amount starting from specific index of a row so

        float candidateCell[HIDDEN_SIZE];
        int hiddensizeTimes2 = hiddenSize*2;
        int hiddensizeTimes3 = hiddenSize*3;

        for(int i = 0; i < hiddenSize; i++)
        {
            inputGate[i] = sigmoid( dotCommon(weightsW[i], VecEmbdsInput[h], hiddenSize) + dotCommon(weightsU[i], hiddenVector, hiddenSize) + bias[i]);
            forgetGate[i] = sigmoid( dotCommon(weightsW[i]+hiddenSize, VecEmbdsInput[h], hiddenSize) + dotCommon(weightsU[i]+hiddenSize, hiddenVector, hiddenSize) + bias[hiddenSize + i]);
            candidateCell[i] = tanh( dotCommon(weightsW[i]+hiddensizeTimes2, VecEmbdsInput[h], hiddenSize) + dotCommon(weightsU[i]+hiddensizeTimes2, hiddenVector, hiddenSize) + bias[hiddensizeTimes2 + i]);
            outputGate[i] = sigmoid( dotCommon(weightsW[i]+hiddensizeTimes3, VecEmbdsInput[h], hiddenSize) + dotCommon(weightsU[i]+hiddensizeTimes3, hiddenVector, hiddenSize) + bias[hiddensizeTimes3 + i]);
        
            cellActivation[i] = (forgetGate[i] * cellActivation[i]) + (inputGate[i] * candidateCell[i]);

            hiddenVector[i] = outputGate[i] * tanh(cellActivation[i]);
        }
    }
    return;
};

void KerasBidiLSTM::forward(float (*VecEmbdsInput)[LSTM_INPUT_SIZE], float* output){ // outputBid is declared in main as [hiddensize*2]
    forwardCell.forward(VecEmbdsInput);

    int VecEmbdsSize = WORD_NUMBER; // NUMBER OF WORDS passed limit, aka VecEmdsInput number of rows

    float VecEmbdsReversed[WORD_NUMBER][LSTM_INPUT_SIZE]; //creating a reversed copy to increase parallelism in future. Costs extra memory
    
    for (int i = VecEmbdsSize - 1, k = 0; i >= 0; i--, k++) {
        for (int j = 0; j < hiddenSizeBid; j++) {
            VecEmbdsReversed[k][j] = VecEmbdsInput[i][j];
        }
    }
    backwardCell.forward(VecEmbdsReversed);

    for(int i = 0; i < hiddenSizeBid; i++){
        output[i] = forwardCell.outputGate[i];
    }
    
    for(int i = hiddenSizeBid; i< hiddenSizeBid*2; i++){
        output[i] = backwardCell.outputGate[i];
    }
    return;
}  