#include "EmbeddingLayer.h"

void EmbeddingLayer::forward(float* WordIndexEmbeddings, float (*output)[EMBED_OUT_SIZE]){

    for(int i =0; i< WORD_NUMBER; i++){ // IF WORD_NUMBER IS NOT CORRECT A SEGFAULT WILL OCCUR
        int RowBeingCopied = WordIndexEmbeddings[i];
        if (RowBeingCopied < 0 || RowBeingCopied >= VocabSize) {continue;} // out of bounds, shouldn't really happen 
        for(int j =0; j< OutSize; j++){
            output[i][j] = weights[RowBeingCopied][j];
        }
    }

    return ;
};
