#include "EmbeddingLayer.h"
#include "KerasBidiLSTM.h"
#include "SmallDense.h"
#include "BigDense.h"

#include <iostream>
#include <chrono>

int main(){
    std::cout << "Program started.\n";

    //Initialize. Set weights and their dimensions in the .h file
    EmbeddingLayer EmbedLayer;
    KerasBidiLSTM BidirLSTMLayer;
    BigDense BigDense;
    SmallDense SmallDense;

    //Jason's part goes here
    //placeholder for now
    float JasonsPartOutput[20] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1325, 282};

    float output1[20][64];
    float output2[128];
    float output3[64];
    float finalOutput[3];

    auto start = std::chrono::high_resolution_clock::now();
    long double clkstart = clock();
    EmbedLayer.forward(JasonsPartOutput, output1);
    BidirLSTMLayer.forward(output1, output2); //input, output as parameters
    BigDense.forwardRelu(output2, output3);
    SmallDense.forwardSoftmax(output3, finalOutput);
    long double clkend = clock();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << clkend-clkstart << "\n" << duration.count() << "\n";

    std::cout << "1:" << finalOutput[0] << "    2:" << finalOutput[1] << "    3:" << finalOutput[2];


    return 0;
}
