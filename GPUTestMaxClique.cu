//GPUTestMaxClique.cu

#include <stdio.h>
#include "cliqueMerging.cuh"

int main(int argc, char** argv)
{
    cudaError_t retval = cudaSuccess;
    struct clique_node* cliques = new clique_node();
    retval = findMaxCliquesGPU("test max clique on gpu", argc, argv, &cliques);

    //printKCliques(cliques);
    delete cliques;
    cudaDeviceReset();

    return 0;
}
