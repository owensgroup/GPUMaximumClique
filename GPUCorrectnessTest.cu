//GPUCorrectnessTest.cu

#include <stdio.h>
#include "cliqueMerging.cuh"

int main(int argc, char** argv)
{
    //brock200_2.mtx test
    int argCount = argc + 4;
    char** commandLine = new char*[argCount];
    for(int i = 0; i < argc; i++){
        commandLine[i] = argv[i];
    }
    char graphFormat[] = "market";
    commandLine[argc] = graphFormat;
    char graphFile[] = "--graph-file=./brock200_2.mtx";
    commandLine[argc + 1] = graphFile;
    char ordering[] = "--order_candidates=false";  //to make sure vertices in output are in same order for correctness check
    commandLine[argc + 2] = ordering;
    char orientation[] = "--orientation=index";    //to make sure vertices in output are in same order
    commandLine[argc + 3] = orientation;

    cudaError_t retval = cudaSuccess;
    struct clique_node* cliques = (struct clique_node*) malloc(sizeof(struct clique_node));
    retval = findMaxCliquesGPU("validation test for max clique on gpu", argCount, commandLine, &cliques);

    //Check result:
    int k = cliques->k;
    long long unsigned int numCliques = cliques->numVertices;
    if(k != 12 || numCliques != 1){
        printf("INCORRECT OUTPUT for brock200_2.\n%llu %i-cliques found.\nCorrect output is 1 12-clique.\n", numCliques, k);
    }
    else{
        unsigned int* brockOutput = readClique(cliques, 12, 0);
        unsigned int groundTruth[12] = {26, 47, 54, 69, 104, 119, 120, 134, 144, 148, 157, 182};
        bool correctResult = true;
        for(int i = 0; i < 12; i++){
            if(brockOutput[i] != groundTruth[i]) correctResult = false;
        }
        if(correctResult == true){
            printf("Correct maximum clique found for brock200_2 dataset.\n");
        }
        else{
            printf("INCORRECT RESULT for brock200_2.\n");
            printf("Correct output: 26, 47, 54, 69, 104, 119, 120, 134, 144, 148, 157, 182\n");
        }
    }

    printKCliques(cliques);
    delete cliques;
    delete commandLine;
    cudaDeviceReset();

    return 0;
}
