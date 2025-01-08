#include "EmbeddingLayer.h"

EmbeddingLayer::EmbeddingLayer(int Out, int Vocab) : OutSize(Out), VocabSize(Vocab) {

        // do nothing legacy funct

}

vector<float*> EmbeddingLayer::forward(vector<int> WordIndexEmbeddings){
    vector<float*> embeddingWeights;
    int LengthVectorInput = WordIndexEmbeddings.size();
    // Allocate memory for embeddings to ensure contiguous allocation of memory
    for (int i = 0; i < LengthVectorInput ; ++i) {
        int index = WordIndexEmbeddings[i];
        if (index < 0 || index >= VocabSize) {continue;} // out of bounds 

        float* embeddingRow = new float[OutSize]; // Allocate memory for embedding
        embeddingWeights.push_back(embeddingRow); // add to vector
    }

    // Fill previously allocated matrix with corresponding weights of rows E[n] = Outsize floats arrays
    for (int i = 0; i < LengthVectorInput; ++i) {
        int index = WordIndexEmbeddings[i];
        if (index < 0 || index >= VocabSize) {continue;} // out of bounds

        // Copy the embedding weights from the weights matrix
        for (int j = 0; j < OutSize; ++j) {
            embeddingWeights[i][j] = weights[index][j];
        }
    }

    return embeddingWeights;
};
