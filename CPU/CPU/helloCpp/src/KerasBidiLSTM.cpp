#include "KerasBidiLSTM.h"


LSTMCellForward::LSTMCellForward(int inputSize , int hiddenSize)
{
    this->inputSize = inputSize;
    this->hiddenSize = hiddenSize;

    for (int i =0; i< hiddenSize; i++){
        inputGate[i] = 0.0f;
        forgetGate[i] = 0.0f;
        outputGate[i] = 0.0f;
        cellActivation[i] = 0.0f;
        hiddenVector[i] = 0.0f;
        candidateCell[i] = 0.0f;
    }
}

//returns a singular 128 length float vector
float* LSTMCellForward::forward( vector<float*> VecEmbdsInput){
    // Implement forward pass using LSTM equations
    int iterationNum = VecEmbdsInput.size();
    for(int h = 0; h < iterationNum; h++){
        // so for these following dots Im gonna need to operate for specific amount starting from specific index of a row so
        for(int i = 0; i < hiddenSize; i++){
            forgetGate[i] = sigmoid( dot(weightsU[i], hiddenVector, hiddenSize) + dot(weightsW[i], VecEmbdsInput[h], hiddenSize) + bias[hiddenSize + i]);
        }
    
        for(int i = 0; i < hiddenSize; i++){
            inputGate[i] = sigmoid( dot(weightsU[i], hiddenVector, 0) + dot(weightsW[i], VecEmbdsInput[h], 0) + bias[i]);
        }

        int SavesFloatMuls = hiddenSize*2;  //state it here to avoid calculating it every time I pass the index maybe prematurely optimized 
        for(int i = 0; i < hiddenSize; i++){
            candidateCell[i] = tanh( dot(weightsU[i], hiddenVector, SavesFloatMuls) + dot(weightsW[i], VecEmbdsInput[h], SavesFloatMuls) + bias[SavesFloatMuls + i]);
        }

        for(int i = 0; i < hiddenSize; i++){
            cellActivation[i] = (forgetGate[i] * cellActivation[i]) + (inputGate[i] * candidateCell[i]);
        }

        SavesFloatMuls = hiddenSize*3;
        for(int i = 0; i < hiddenSize; i++){
            outputGate[i] = sigmoid( dot(weightsU[i], hiddenVector, SavesFloatMuls) + dot(weightsW[i], VecEmbdsInput[h], SavesFloatMuls) + bias[SavesFloatMuls + i]);
        }

        for(int i = 0; i < hiddenSize; i++){
            hiddenVector[i] = outputGate[i] * tanh(cellActivation[i]);
        }
    }
    return outputGate;
};

LSTMCellBackward::LSTMCellBackward(int inputSize , int hiddenSize)
{
    this->inputSize = inputSize;
    this->hiddenSize = hiddenSize;

    for (int i =0; i< hiddenSize; i++){
        inputGate[i] = 0.0f;
        forgetGate[i] = 0.0f;
        outputGate[i] = 0.0f;
        cellActivation[i] = 0.0f;
        hiddenVector[i] = 0.0f;
        candidateCell[i] = 0.0f;
    }
}

//returns a singular 128 length float vector
float* LSTMCellBackward::forward( vector<float*> VecEmbdsInput){
    // Implement forward pass using LSTM equations
    int iterationNum = VecEmbdsInput.size();
    for(int h = 0; h < iterationNum; h++){
        // so for these following dots Im gonna need to operate for specific amount starting from specific index of a row so
        for(int i = 0; i < hiddenSize; i++){
            forgetGate[i] = sigmoid( dot(weightsU[i], hiddenVector, hiddenSize) + dot(weightsW[i], VecEmbdsInput[h], hiddenSize) + bias[hiddenSize + i]);
        }
    
        for(int i = 0; i < hiddenSize; i++){
            inputGate[i] = sigmoid( dot(weightsU[i], hiddenVector, 0) + dot(weightsW[i], VecEmbdsInput[h], 0) + bias[i]);
        }
        
        int SavesFloatMuls = hiddenSize*2;  //state it here to avoid calculating it every time I pass the index maybe prematurely optimized 
        for(int i = 0; i < hiddenSize; i++){
            candidateCell[i] = tanh( dot(weightsU[i], hiddenVector, SavesFloatMuls) + dot(weightsW[i], VecEmbdsInput[h], SavesFloatMuls) + bias[SavesFloatMuls + i]);
        }

        for(int i = 0; i < hiddenSize; i++){
            cellActivation[i] = (forgetGate[i] * cellActivation[i]) + (inputGate[i] * candidateCell[i]);
        }

        SavesFloatMuls = hiddenSize*3;
        for(int i = 0; i < hiddenSize; i++){
            outputGate[i] = sigmoid( dot(weightsU[i], hiddenVector, SavesFloatMuls) + dot(weightsW[i], VecEmbdsInput[h], SavesFloatMuls) + bias[SavesFloatMuls + i]);
        }

        for(int i = 0; i < hiddenSize; i++){
            hiddenVector[i] = outputGate[i] * tanh(cellActivation[i]);
        }
    }
    return outputGate;
};

KerasBidiLSTM::KerasBidiLSTM(int inSize, int hidSize){
    inputSizeBid = inSize;
    hiddenSizeBid = hidSize;
};

KerasBidiLSTM::~KerasBidiLSTM(){
}


void KerasBidiLSTM::forward(vector<float*> VecEmbdsInput, float* outputBid){ // outputBid is declared in main as [hiddensize*2]
    float *forwardResult = forwardCell.forward(VecEmbdsInput);

    int VecEmbdsSize = VecEmbdsInput.size();

    std::vector<float*> VecEmbdsReversed(VecEmbdsSize); //creating a reversed copy to increase parallelism in future. Costs extra memory
    for (int i = 0; i < VecEmbdsSize; ++i) {
        VecEmbdsReversed[i] = VecEmbdsInput[VecEmbdsSize - 1 - i]; 
    }
    float *backwardsResult = backwardCell.forward(VecEmbdsReversed);

    for(int i = 0; i < hiddenSizeBid; i++){
        outputBid[i] = forwardResult[i];
    }
    for(int i = hiddenSizeBid; i< hiddenSizeBid*2; i++){
        outputBid[i] = backwardsResult[i];
    }
    return;
}  