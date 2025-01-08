#include "EmbeddingLayer.h"
#include "KerasBidiLSTM.h"
#include "SmallDense.h"
#include "BigDense.h"
#include "train.txt"
#include "dev.txt"
#include "test.txt"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <vector>
#include <string>
//#include <sys/sysctl.h>
//#include <unistd.h>
//#include <mach/mach.h>
#include <algorithm>
#include <chrono>


using namespace std;


const int TRAIN_LENGTH = 489;
const int DEV_LENGTH = 22621;
const int TEST_LENGTH = 6000;

int main(){
    
    cout << "Program started.\n";
    //So first let's declare all layers we need.
    //This was built off of a more general CPU model that would declare EmbedLayer(64, 1000)
    //so Layer constructions recieves integers as a legacy feature but it is primarily 
    //for human readability
    EmbeddingLayer EmbedLayer(64, 10000);
    KerasBidiLSTM BidirLSTMLayer(64, 64);
    BigDense BigDense(128, 64);
    SmallDense SmallDense(64, 3);

    //Jason's part goes here
    //placeholder for now

    
    // I'm going to assume that I don't need to reinstantiate the vectors for every trial. 
    // This is to avoid having to deal with memory management
    float output2[128];
    float output3[64];
    float finalOutput[3];


    // Training Set
    double correctTrials = 0;
    long double averageTime = 0.0;
    unsigned long int averageClock = 0;
    long double averageThroughput = 0.0;
    long double averageLatency = 0.0;
    // Three thousandths are defined for postive, neutral, and negative respectively
    int whole, thousandths1, thousandths2, thousandths3;
    for(int i = 0; i < TRAIN_LENGTH; i++)
    {
        chrono::nanoseconds start = chrono::duration_cast<chrono::nanoseconds>(chrono::system_clock::now().time_since_epoch());
        long double clkstart = clock();
        vector<float*> output1 = EmbedLayer.forward(JasonsPartOutputTrain[i]);
        BidirLSTMLayer.forward(output1, output2); //input, output as parameters
        BigDense.forwardRelu(output2, output3);
        SmallDense.forwardSoftmax(output3, finalOutput);
        chrono::nanoseconds end = chrono::duration_cast<chrono::nanoseconds>(chrono::system_clock::now().time_since_epoch());
        long double clkend = clock();
        auto duration = chrono::duration_cast<chrono::duration<long double>>(end - start);

        averageClock += (clkend - clkstart);
        averageTime += duration.count();
        //averageThroughput += ((10000.00 + 64.00 + 64.00 + 64.00)/duration.count());
        //averageLatency += (1 / ((10000.00 + 64.00 + 64.00 + 64.00)/duration.count()));


        whole = finalOutput[0];
        thousandths1 = (finalOutput[0] - whole) * 1000000;
        whole = finalOutput[1];
        thousandths2 = (finalOutput[1] - whole) * 1000000;
        whole = finalOutput[2];
        thousandths3 = (finalOutput[2] - whole) * 1000000;
        // Assumes the values won't be the same and hopefully they aren't
        if(thousandths1 > thousandths2 && thousandths1 > thousandths3)
        {
            JasonsPartOutputTrain[i].push_back(0);
        }
        else if(thousandths2 > thousandths1 && thousandths2 > thousandths3)
        {
            JasonsPartOutputTrain[i].push_back(1);
        }
        else if(thousandths3 > thousandths1 && thousandths3 > thousandths2)
        {
            JasonsPartOutputTrain[i].push_back(2);
        }

    }
    
    averageThroughput = TRAIN_LENGTH/averageTime;
    averageLatency = (1 / averageThroughput);
    for(int i = 0; i < TRAIN_LENGTH; i++)
    {
        if(JasonsPartOutputTrain[i][20] == correctResultsTrain[i])
        {
            correctTrials = correctTrials + 1;
        }
    }

    cout << "Training Results: " << endl;
    cout << "Number of trials: " << TRAIN_LENGTH << " trials." << endl;
    cout << "Average time in seconds: " << fixed << setprecision(5) << (averageTime/TRAIN_LENGTH) << " seconds." << endl;
    cout << "Average number of clock cycles: " << (averageClock/TRAIN_LENGTH) << " cycles." << endl;
    cout << "Average throughput: " << fixed << setprecision(2) << averageThroughput << " samples per second." << endl;
    cout << "Average latency: " << fixed << setprecision(8) << averageLatency << " seconds per sample" << endl;
    cout << "Overall Accuracy: " << fixed << setprecision(2) << ((correctTrials/TRAIN_LENGTH)*100) << "%" << endl;




    // Developer Set


    double correctTrialsDev = 0;
    long double averageTimeDev = 0.0;
    unsigned long int averageClockDev = 0;
    long double averageThroughputDev = 0.0;
    long double averageLatencyDev = 0.0;
    // Three thousandths are defined for postive, neutral, and negative respectively
    int wholeDev, thousandths1Dev, thousandths2Dev, thousandths3Dev;
    for(int i = 0; i < DEV_LENGTH; i++)
    {
        chrono::nanoseconds start = chrono::duration_cast<chrono::nanoseconds>(chrono::system_clock::now().time_since_epoch());
        long double clkstart = clock();
        vector<float*> output1 = EmbedLayer.forward(JasonsPartOutputDev[i]);
        BidirLSTMLayer.forward(output1, output2); //input, output as parameters
        BigDense.forwardRelu(output2, output3);
        SmallDense.forwardSoftmax(output3, finalOutput);
        chrono::nanoseconds end = chrono::duration_cast<chrono::nanoseconds>(chrono::system_clock::now().time_since_epoch());
        long double clkend = clock();
        auto duration = chrono::duration_cast<chrono::duration<long double>>(end - start);

        averageClockDev += (clkend - clkstart);
        averageTimeDev += duration.count();
        //averageThroughputDev += ((10000.00 + 64.00 + 64.00 + 64.00)/duration.count());
        //averageLatencyDev += (1 / ((10000.00 + 64.00 + 64.00 + 64.00)/duration.count()));


        wholeDev = finalOutput[0];
        thousandths1Dev = (finalOutput[0] - wholeDev) * 1000000;
        wholeDev = finalOutput[1];
        thousandths2Dev = (finalOutput[1] - wholeDev) * 1000000;
        wholeDev = finalOutput[2];
        thousandths3Dev = (finalOutput[2] - wholeDev) * 1000000;
        // Assumes the values won't be the same and hopefully they aren't
        if(thousandths1Dev > thousandths2Dev && thousandths1Dev > thousandths3Dev)
        {
            JasonsPartOutputDev[i].push_back(0);
        }
        else if(thousandths2Dev > thousandths1Dev && thousandths2Dev > thousandths3Dev)
        {
            JasonsPartOutputDev[i].push_back(1);
        }
        else if(thousandths3Dev > thousandths1Dev && thousandths3Dev > thousandths2Dev)
        {
            JasonsPartOutputDev[i].push_back(2);
        }

    }
    
    averageThroughputDev = DEV_LENGTH/averageTimeDev;
    averageLatencyDev = (1 / averageThroughputDev);

    for(int i = 0; i < DEV_LENGTH; i++)
    {
        if(JasonsPartOutputDev[i][20] == correctResultsDev[i])
        {
            correctTrialsDev = correctTrialsDev + 1;
        }
    }

    cout << "Development Results: " << endl;
    cout << "Number of trials: " << DEV_LENGTH << " trials." << endl;
    cout << "Average time in seconds: " << fixed << setprecision(5) << (averageTimeDev/DEV_LENGTH) << " seconds." << endl;
    cout << "Average number of clock cycles: " << (averageClockDev/DEV_LENGTH) << " cycles." << endl;
    cout << "Average throughput: " << fixed << setprecision(2) << averageThroughputDev << " samples per second." << endl;
    cout << "Average latency: " << fixed << setprecision(8) << averageLatencyDev << " seconds per sample" << endl;
    cout << "Overall Accuracy: " << fixed << setprecision(2) << ((correctTrialsDev/DEV_LENGTH)*100) << "%" << endl;





    // Test set

    double correctTrialsTest = 0;
    long double averageTimeTest = 0.0;
    unsigned long int averageClockTest = 0;
    long double averageThroughputTest = 0.0;
    long double averageLatencyTest = 0.0;
    // Three thousandths are defined for postive, neutral, and negative respectively
    int wholeTest, thousandths1Test, thousandths2Test, thousandths3Test;
    for(int i = 0; i < TEST_LENGTH; i++)
    {
        chrono::nanoseconds start = chrono::duration_cast<chrono::nanoseconds>(chrono::system_clock::now().time_since_epoch());
        long double clkstart = clock();
        vector<float*> output1 = EmbedLayer.forward(JasonsPartOutputTest[i]);
        BidirLSTMLayer.forward(output1, output2); //input, output as parameters
        BigDense.forwardRelu(output2, output3);
        SmallDense.forwardSoftmax(output3, finalOutput);
        chrono::nanoseconds end = chrono::duration_cast<chrono::nanoseconds>(chrono::system_clock::now().time_since_epoch());
        long double clkend = clock();
        auto duration = chrono::duration_cast<chrono::duration<long double>>(end - start);

        averageClockTest += (clkend - clkstart);
        averageTimeTest += duration.count();
        //averageThroughputTest += ((10000.00 + 64.00 + 64.00 + 64.00)/duration.count());
        //averageLatencyTest += (1 / ((10000.00 + 64.00 + 64.00 + 64.00)/duration.count()));


        wholeTest = finalOutput[0];
        thousandths1Test = (finalOutput[0] - wholeTest) * 1000000;
        wholeTest = finalOutput[1];
        thousandths2Test = (finalOutput[1] - wholeTest) * 1000000;
        wholeTest = finalOutput[2];
        thousandths3Test = (finalOutput[2] - wholeTest) * 1000000;
        // Assumes the values won't be the same and hopefully they aren't
        if(thousandths1Test > thousandths2Test && thousandths1Test > thousandths3Test)
        {
            JasonsPartOutputTest[i].push_back(0);
        }
        else if(thousandths2Test > thousandths1Test && thousandths2Test > thousandths3Test)
        {
            JasonsPartOutputTest[i].push_back(1);
        }
        else if(thousandths3Test > thousandths1Test && thousandths3Test > thousandths2Test)
        {
            JasonsPartOutputTest[i].push_back(2);
        }

    }
    
    averageThroughputTest = TEST_LENGTH/averageTimeTest;
    averageLatencyTest = (1 / averageThroughputTest);
    for(int i = 0; i < TEST_LENGTH; i++)
    {
        if(JasonsPartOutputTest[i][20] == correctResultsTest[i])
        {
            correctTrialsTest = correctTrialsTest + 1;
        }
    }

    cout << "Test Results: " << endl;
    cout << "Number of trials: " << TEST_LENGTH << " trials." << endl;
    cout << "Average time in seconds: " << fixed << setprecision(5) << (averageTimeTest/TEST_LENGTH) << " seconds." << endl;
    cout << "Average number of clock cycles: " << (averageClockTest/TEST_LENGTH) << " cycles." << endl;
    cout << "Average throughput: " << fixed << setprecision(2) << averageThroughputTest << " samples per second." << endl;
    cout << "Average latency: " << fixed << setprecision(8) << averageLatencyTest << " seconds per sample" << endl;
    cout << "Overall Accuracy: " << fixed << setprecision(2) << ((correctTrialsTest/TEST_LENGTH)*100) << "%" << endl;


    return 0;
}



