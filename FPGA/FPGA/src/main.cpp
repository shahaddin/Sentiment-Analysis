#include "EmbeddingLayer.h"
#include "KerasBidiLSTM.h"
#include "SmallDense.h"
#include "BigDense.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>

#define TEST_TRIALS 6000

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

    long double TotalClk{};
    std::chrono::duration<double> TotalDur{};
    long double accuracy{};

    std::ifstream file("/Users/villaketh/Desktop/Inside/src/test.csv");

    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return 1;
    }

    // Read data from the CSV file
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int value;
        int i =0;
        while (iss >> value && i < 20) {
            JasonsPartOutput[i] = value;
            if (iss.peek() == ',')
                iss.ignore();
            i++;
        }

        auto start = std::chrono::high_resolution_clock::now();
        long double clkstart = clock();
        EmbedLayer.forward(JasonsPartOutput, output1);
        BidirLSTMLayer.forward(output1, output2); //input, output as parameters
        BigDense.forwardRelu(output2, output3);
        SmallDense.forwardSoftmax(output3, finalOutput);
        long double clkend = clock();

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        
        TotalClk += (clkend-clkstart);
        TotalDur += duration;

        float MaxValHere{};
        int answer;
        for(int i =0; i < 3; i++){
            if(MaxValHere < finalOutput[i]){
                MaxValHere = finalOutput[i];
                answer = i;
            }
        }
        iss >> value;
        if( value == answer){
            accuracy++;
        }
        
    }
    
    long double averageThroughput = 6000.00/TotalDur.count();
    long double averageLatency = 1/averageThroughput;
    TotalClk /= 6000;
    TotalDur /= 6000;
    accuracy /= 6000;


    std::cout << TotalClk << "Average clks.     Duration Avg:" << TotalDur.count() << "    acc:" << accuracy << std::endl;
    std::cout << "Average throughput: " << std::fixed << std::setprecision(2) << averageThroughput << " samples per second." << std::endl;
    std::cout << "Average latency: " << std::fixed << std::setprecision(8) << averageLatency << " seconds per sample" << std::endl;
    return 0;
}
