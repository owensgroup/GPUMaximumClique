//cliqueMerging.cu
//Clique merging/breadth-first-search-style implementation for solving maximum clique

#include <stdio.h>
#include <cub/cub.cuh>

#include <gunrock/gunrock.h>
#include <gunrock/util/test_utils.cuh>
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/kcore/kcore_enactor.cuh>
#include <gunrock/app/kcore/kcore_app.cu>

#include "cliqueMerging.cuh"
#include "jsonwriter.cuh"

#include <cuda_profiler_api.h>

typedef typename gunrock::app::TestGraph<unsigned int, unsigned int, unsigned int,
            gunrock::graph::HAS_CSR>GraphT;     //VertexT, SizeT, ValueT are uint
typedef typename GraphT::CsrT CsrT;


// Functor type for selecting values greater than some criteria
struct GreaterThan
{
    int compare;
    CUB_RUNTIME_FUNCTION __forceinline__
    GreaterThan(int compare) : compare(compare) {}
    CUB_RUNTIME_FUNCTION __forceinline__
    bool operator()(const int &a) const {
        return (a > compare);
    }
};

clique_node::clique_node(){
    numVertices = 0llu;
    k = 0;
    vertexIDs = NULL;
    sublistIDs = NULL;
    previous = NULL;
}

clique_node::~clique_node(){

}

time_breakdown::time_breakdown(){
    total = 0.0f;
    heuristic = 0.0f;
    kcore = 0.0f;
    presort = 0.0f;
    two_cliques = 0.0f;
    postsort = 0.0f;
    total_preproc = 0.0f;
    dfs = 0.0f;
    bfs = 0.0f;
}

time_breakdown::~time_breakdown(){

}

bfs_loop_breakdown::bfs_loop_breakdown(){
    count = 0.0f;
    scan_alloc = 0.0f;
    merge = 0.0f;
}

bfs_loop_breakdown::~bfs_loop_breakdown(){

}

dfs_loop_breakdown::dfs_loop_breakdown(){
    find_window = 0.0f;
    count = 0.0f;
    scan_alloc = 0.0f;
    merge = 0.0f;
}

dfs_loop_breakdown::~dfs_loop_breakdown(){

}

__host__ cudaError_t CUDAErrorCheck()
{
    cudaError_t errSync = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess) {
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        cudaDeviceReset();
        return errSync;
    }
    if (errAsync != cudaSuccess) {
	printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
        cudaDeviceReset();
        return errAsync;
    }
    return cudaSuccess;
}

__host__ cudaError_t insertNewHeadNode(struct clique_node** oldHead, long long unsigned int size, int k)
{
    /*  Allocates memory for a new clique list node at the head of the clique list. The number of vertices
     *  in this node is size and k is the level of the clique list, which also represents the size of the 
      * clique. The pointer oldHead is updated to point to the new head node.   */
    cudaError_t retval = cudaSuccess;
    struct clique_node* newNode = new clique_node();
    newNode->numVertices = size;
    newNode->k = k;
    unsigned int *d_vertices, *d_sublists;
    GUARD_CU(cudaMalloc((void**) &d_vertices, size * sizeof(unsigned int)));
    GUARD_CU(cudaMalloc((void**) &d_sublists, size * sizeof(unsigned int)));
    newNode->vertexIDs = d_vertices;
    newNode->sublistIDs = d_sublists;
    newNode->previous = *(oldHead);
    *oldHead = newNode;
    return retval;
}

__device__ __host__ bool areConnected(CsrT graph, int src, int dest)
{
    /*  Check whether vertices src and dest are connected.  */
    //get the neighbor list start & length for src
    int listStart = graph.CsrT::GetNeighborListOffset(src);
    int listLength = graph.CsrT::GetNeighborListLength(src);
    //search within that list for dest (binary search)
    int destIndex = gunrock::util::BinarySearch(dest, graph.column_indices, listStart, (listStart + listLength - 1));

    return (graph.column_indices[destIndex] == dest);
}

struct IsNeighbor
{
    int src;
    CsrT graph;
    __host__ __device__ __forceinline__ IsNeighbor(CsrT inputGraph, int maxNode) : graph(inputGraph), src(maxNode) {}

    __host__ __device__  __forceinline__ bool operator()(const int &dest) const {
        return areConnected(graph, src, dest);
    }
};

__host__ void printSublists(struct clique_node* currentNode)
{
    /*  Prints all of the sublists in the node of the clique list pointed to by currentNode.  */
    //copy necessary info to CPU
    long long unsigned int size = currentNode->numVertices;
    unsigned int* h_sublistIDs = new unsigned int[size];
    cudaMemcpy(h_sublistIDs, currentNode->sublistIDs, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int* h_vertexIDs = new unsigned int[size];
    cudaMemcpy(h_vertexIDs, currentNode->vertexIDs, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //print the sublists
    for(long long unsigned int i = 0; i < size; i++) {
        unsigned int sublist = h_sublistIDs[i];
        printf("%u:\t", sublist);
        long long unsigned int j = 0;
        while(((i+j) < size) && (sublist == h_sublistIDs[i+j])) {
            printf("%u, ", h_vertexIDs[i+j]);
            j++;
        }
        printf("\n");
        i += (j-1);
    }
}

__host__ unsigned int* readClique(struct clique_node* k_cliques, int cliqueSize, long long unsigned int index)
{
    /*  Prints all of vertices in the index-th clique. k_cliques should point to the head node of 
     *  the clique list.  */
    unsigned int* clique = new unsigned int [cliqueSize];
    while((cliqueSize > 1) && (k_cliques != NULL)) {
        cliqueSize--;
        cudaMemcpy((clique + cliqueSize), (k_cliques->vertexIDs + index), sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&(index), (k_cliques->sublistIDs + index), sizeof(unsigned int), cudaMemcpyDeviceToHost);
        k_cliques = k_cliques->previous;
    }
    //final vertex is stored in the sublist ID for the tail node
    if (k_cliques == NULL) {
        clique[0] = index;
    }
    return clique;
}

__host__ void printKCliques(struct clique_node* k_cliques)
{
    /*  Prints all of vertices in all k-cliques using the readCliques function. k_cliques should point 
     *  to the head node of the clique list.  */
    int k = k_cliques->k;
    printf("%i-cliques:\t", k);
    for (long long unsigned int i = 0; i < k_cliques->numVertices; i++) {
        unsigned int* clique = readClique(k_cliques, k, i);
        for (int j = 0; j < k; j++) {
            printf("%i, ", clique[j]);
        }
        printf("\n");
    }
}

__host__ void printKCliques_preempted(struct clique_node* k_cliques)
{
    /*  For instances where the maximum clique can be found without performing all k iterations of the 
     *  main algorithm. In this case, all vertices in the final node of the clique list are members of
     *  the maximum clique. Prints all of vertices in all k-cliques. k_cliques should point to the head 
     *  node of the clique list.  */
     //do readClique for first index (or any index), then print vertices in final node
    int k = k_cliques->k;
    printf("%i-clique:\t", (k + k_cliques->numVertices - 1));
    //read previous clique nodes for first index only
    unsigned int* clique = readClique(k_cliques, k, 0);
    for (int j = 0; j < k; j++) {
        printf("%i, ", clique[j]);
    }
    //print rest of vertices in final node
    unsigned int* vertices = new unsigned int [k_cliques->numVertices];
    cudaMemcpy(vertices, k_cliques->vertexIDs, k_cliques->numVertices * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for (long long unsigned int i = 1; i < k_cliques->numVertices; i++) {
        printf("%i, ", vertices[i]);
    }
    printf("\n");
}

__global__ void countTwoCliques(CsrT graph, unsigned int numVertices, unsigned int* d_vertexLabels, unsigned int* d_vertexDegrees, unsigned int* d_filterThresholds, unsigned int w, bool byDegree, unsigned int* counts, char* flags)
{
    /*  Returns the number of 2-cliques (edges) for each source vertex, keeping only
     *  one edge from each undirected pair, based on either index or degree orientation, and
     *  where both vertices' threshold values are greater than the current lower bound max clique size.    */
    long long unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;

    unsigned int src = d_vertexLabels[idx];
    if (d_filterThresholds[src] < (w - 1)) return;
    int listStart = graph.CsrT::GetNeighborListOffset(src);
    int numNeighbors = d_vertexDegrees[src];
    int listEnd = listStart + numNeighbors;
    unsigned int validCount = 0;
    if (!byDegree) {    //select edge from each undirected pair based on vertices' indices
        for (int e = listStart; e < listEnd; e++) {
            unsigned int dest = graph.CsrT::GetEdgeDest(e);
            if (src < dest) {   //index orientation
                if (!(d_filterThresholds[dest] < (w - 1))) {
                    validCount++;
                }
            }
        }
    }
    if (byDegree) { //select edge from each undirected pair based on vertices' degrees
        for (int e = listStart; e < listEnd; e++) {
            unsigned int dest = graph.CsrT::GetEdgeDest(e);
            int numNeighborsDest = d_vertexDegrees[dest];
            if ((numNeighbors < numNeighborsDest) || ((numNeighbors == numNeighborsDest) && (src < dest))) { //degree orientation
                if (!(d_filterThresholds[dest] < (w - 1))) {
                    validCount++;
                }
            }
        }
    }
    
    if (validCount >= (w - 1)) {
        counts[idx] = validCount;
        flags[idx] = 1;
    }
}

__global__ void outputTwoCliques_noReorder(CsrT graph, unsigned int numSublists, unsigned int* d_vertexLabels_filtered, unsigned int* d_startIndices, unsigned int* d_vertexDegrees, unsigned int* d_filterThresholds, unsigned int w, bool byDegree, struct clique_node d_two_cliques)
{
    /*  Outputs the 2-cliques (edges) for each source vertex, keeping only one
     *  edge from each undirected pair, based on either index or degree, and
     *  where both vertices' threshold values are greater than the current lower bound.    */
    long long unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSublists) return;

    unsigned int src = d_vertexLabels_filtered[idx];
    unsigned int startIndex = d_startIndices[idx];
    int listStart = graph.CsrT::GetNeighborListOffset(src);
    int numNeighbors = d_vertexDegrees[src];
    int listEnd = listStart + numNeighbors;
    unsigned int validCount = 0;
    if (!byDegree) {    //select edge from each undirected pair based on vertices' indices
        for (int e = listStart; e < listEnd; e++) {
            unsigned int dest = graph.CsrT::GetEdgeDest(e);
            if (src < dest) {   //index orientation
                if (!(d_filterThresholds[dest] < (w - 1))) {
                    d_two_cliques.sublistIDs[startIndex + validCount] = src;
                    d_two_cliques.vertexIDs[startIndex + validCount] = dest;
                    validCount++;
                }
            }
        }
    }
    if (byDegree) { //select edge from each undirected pair based on vertices' degrees
        for (int e = listStart; e < listEnd; e++) {
            unsigned int dest = graph.CsrT::GetEdgeDest(e);
            int numNeighborsDest = d_vertexDegrees[dest];
            if ((numNeighbors < numNeighborsDest) || ((numNeighbors == numNeighborsDest) && (src < dest))) { //degree orientation
                if (!(d_filterThresholds[dest] < (w - 1))) {
                    d_two_cliques.sublistIDs[startIndex + validCount] = src;
                    d_two_cliques.vertexIDs[startIndex + validCount] = dest;
                    validCount++;
                }
            }
        }
    }
}

__global__ void outputTwoCliques_reorder(CsrT graph, unsigned int numSublists, unsigned int* d_vertexLabels_filtered, unsigned int* d_startIndices, unsigned int* d_vertexDegrees, unsigned int* d_filterThresholds, unsigned int w, bool byDegree, struct clique_node d_two_cliques, unsigned int* d_candidateDegrees)
{
    /*  Outputs the 2-cliques (edges) for each source vertex, keeping only one
     *  edge from each undirected pair, based on either index or degree, and
     *  where both vertices' threshold values are greater than the current lower bound.
     *  Also outputs candidate degrees, to be used to sort them in a following step.    */
    long long unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSublists) return;

    unsigned int src = d_vertexLabels_filtered[idx];
    unsigned int startIndex = d_startIndices[idx];
    int listStart = graph.CsrT::GetNeighborListOffset(src);
    int numNeighbors = d_vertexDegrees[src];
    int listEnd = listStart + numNeighbors;
    unsigned int validCount = 0;
    if (!byDegree) {    //select edge from each undirected pair based on vertices' indices
        for (int e = listStart; e < listEnd; e++) {
            unsigned int dest = graph.CsrT::GetEdgeDest(e);
            if (src < dest) {   //index orientation
                if (!(d_filterThresholds[dest] < (w - 1))) {
                    d_two_cliques.sublistIDs[startIndex + validCount] = src;
                    d_two_cliques.vertexIDs[startIndex + validCount] = dest;
                    d_candidateDegrees[startIndex + validCount] = d_vertexDegrees[dest];
                    validCount++;
                }
            }
        }
    }
    if (byDegree) { //select edge from each undirected pair based on vertices' degrees
        for (int e = listStart; e < listEnd; e++) {
            unsigned int dest = graph.CsrT::GetEdgeDest(e);
            int numNeighborsDest = d_vertexDegrees[dest];
            if ((numNeighbors < numNeighborsDest) || ((numNeighbors == numNeighborsDest) && (src < dest))) { //degree orientation
                if (!(d_filterThresholds[dest] < (w - 1))) {
                    d_two_cliques.sublistIDs[startIndex + validCount] = src;
                    d_two_cliques.vertexIDs[startIndex + validCount] = dest;
                    d_candidateDegrees[startIndex + validCount] = numNeighborsDest;
                    validCount++;
                }
            }
        }
    }
}

__host__ cudaError_t setUpTwoCliques(CsrT graph, unsigned int* d_vertexLabels, unsigned int* d_vertexDegrees, unsigned int* d_filterThresholds, unsigned int w, bool orientByDegree, bool& orderCandidates, struct clique_node* d_two_cliques, cudaEvent_t beginPostSort, cudaEvent_t endPostSort, bool& preemptMain)
{
    /*  Returns first node of the clique list struct, filled with a pre-pruned list of 2-cliques (edges) in d_two_cliques.
     *  Edges with src or dest vertices' threshold (degree or core number) values less than the input lower bound
     *  are pre-pruned, and candidate lists with length less than the lower bound are pruned as well.
     *  Inputs vertexDegrees and filterThresholds must be sorted by index, whether or not vertexLabels is sorted by
     *  degree/k-core. Does not output segment info and candidate degrees array for sorting vertices within sublists by degree.    */
    cudaError_t retval = cudaSuccess;
    unsigned int numVertices = graph.nodes;
    //count 2-cliques for each src vertex
    unsigned int* d_cliqueCounts;
    cudaMalloc((void**) &d_cliqueCounts, numVertices * sizeof(unsigned int));
    cudaMemset(d_cliqueCounts, 0, numVertices * sizeof(unsigned int));
    char* d_keepFlags;
    cudaMalloc((void**) &d_keepFlags, numVertices * sizeof(char));
    cudaMemset(d_keepFlags, 0, numVertices * sizeof(char));
    countTwoCliques<<<(numVertices + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(graph, numVertices, d_vertexLabels, d_vertexDegrees, d_filterThresholds, w, orientByDegree, d_cliqueCounts, d_keepFlags);

    //compact out vertices with candidate lists shorter than lower bound
    unsigned int* d_cliqueCounts_filtered;
    cudaMalloc((void**) &d_cliqueCounts_filtered, numVertices * sizeof(unsigned int));
    unsigned int* d_vertexLabels_filtered;
    cudaMalloc((void**) &d_vertexLabels_filtered, numVertices * sizeof(unsigned int));
    unsigned int* d_numSublists;
    cudaMalloc((void**) &d_numSublists, sizeof(unsigned int));
    unsigned int h_numSublists = 0;
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_cliqueCounts, d_keepFlags, d_cliqueCounts_filtered, d_numSublists, numVertices));
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_cliqueCounts, d_keepFlags, d_cliqueCounts_filtered, d_numSublists, numVertices));
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_vertexLabels, d_keepFlags, d_vertexLabels_filtered, d_numSublists, numVertices));
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_vertexLabels, d_keepFlags, d_vertexLabels_filtered, d_numSublists, numVertices));
    cudaMemcpy(&h_numSublists, d_numSublists, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("number of sublists after filtering: %u\n", h_numSublists);

    //prefix sum to find total number of 2-cliques and sublist start indices
    unsigned int* d_indices;
    cudaMalloc((void**) &d_indices, (h_numSublists + 1) * sizeof(unsigned int));
    d_temp_storage = NULL;
    temp_storage_bytes = 0;
    CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_cliqueCounts_filtered, d_indices, (h_numSublists + 1)));
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_cliqueCounts_filtered, d_indices, (h_numSublists + 1)));

    //Allocate memory for neighbor lists and their threshold values
    unsigned int numTwoCliques;
    cudaMemcpy(&numTwoCliques, d_indices + h_numSublists, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if ((h_numSublists == 1) && (numTwoCliques == w - 1)) {
        preemptMain = true;
        orderCandidates = false;
    }

    //set up and allocate memory for 2-cliques
    d_two_cliques->k = 2;
    d_two_cliques->numVertices = numTwoCliques;
    cudaMalloc((void**) &(d_two_cliques->vertexIDs), numTwoCliques * sizeof(unsigned int));
    cudaMalloc((void**) &(d_two_cliques->sublistIDs), numTwoCliques * sizeof(unsigned int));
    d_two_cliques->previous = NULL;

    if (orderCandidates == false) {
        //fill clique node with 2-clique info
        outputTwoCliques_noReorder<<<(h_numSublists + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(graph, h_numSublists, d_vertexLabels_filtered, d_indices, d_vertexDegrees, d_filterThresholds, w, orientByDegree, *d_two_cliques);
    }

    if (orderCandidates == true) {
        //fill clique node with 2-clique info
        unsigned int* d_candidateDegrees;
        cudaMalloc((void**) &d_candidateDegrees, numTwoCliques * sizeof(unsigned int));
        outputTwoCliques_reorder<<<(h_numSublists + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(graph, h_numSublists, d_vertexLabels_filtered, d_indices, d_vertexDegrees, d_filterThresholds, w, orientByDegree, *d_two_cliques, d_candidateDegrees);

        //Sort vertices in each sublist from low to high degree
        cudaEventRecord(beginPostSort);
        unsigned int* d_candidateDegrees_out;
        cudaMalloc((void**) &d_candidateDegrees_out, numTwoCliques * sizeof(unsigned int));
        unsigned int* d_vertexIDs_out;
        cudaMalloc((void**) &d_vertexIDs_out, numTwoCliques * sizeof(unsigned int));
        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        CubDebugExit(cub::DeviceSegmentedSort::SortPairs(d_temp_storage, temp_storage_bytes, d_candidateDegrees, d_candidateDegrees_out, d_two_cliques->vertexIDs, d_vertexIDs_out, numTwoCliques, h_numSublists, d_indices, d_indices + 1));
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        CubDebugExit(cub::DeviceSegmentedSort::SortPairs(d_temp_storage, temp_storage_bytes, d_candidateDegrees, d_candidateDegrees_out, d_two_cliques->vertexIDs, d_vertexIDs_out, numTwoCliques, h_numSublists, d_indices, d_indices + 1));

        unsigned int* tempIDs = d_two_cliques->vertexIDs;
        d_two_cliques->vertexIDs = d_vertexIDs_out;
        cudaFree(tempIDs);
        cudaFree(d_candidateDegrees_out);
        cudaEventRecord(endPostSort);
    }

    cudaFree(d_cliqueCounts);
    cudaFree(d_keepFlags);
    cudaFree(d_cliqueCounts_filtered);
    cudaFree(d_vertexLabels_filtered);
    cudaFree(d_numSublists);
    cudaFree(d_indices);

    return retval;
}

__global__ void findWindowTail(struct clique_node currCliques, unsigned int windowSize, long long unsigned int start, long long unsigned int* tail)
{
    /*  Returns the end of the window for windowed version of clique merging algorithm.
     *  Threads are assigned to one sublistID each, and check if it is the beginning of a sublist.
     *  If yes, use atomic minimum to find the first location where this happens. Return in tail.
     *  Loop over another windowSize chunk of vertices until a sublist start is found.  */
    long long unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (windowSize - 1)) return;
    idx += start;

    unsigned int iter = 0;
    while(tail[0] > (start + (windowSize * iter))) {

        if (idx >= (currCliques.numVertices - 1)) return;

        if(currCliques.sublistIDs[idx] != currCliques.sublistIDs[idx + 1]) {
            atomicMin(tail, idx);
        }

        iter++;
        idx += iter * windowSize;
    }
}

__global__ void countNewCliques(struct clique_node currCliques, CsrT graph, unsigned int w, long long unsigned int* counts)
{
    /*  Returns the number of cliques that can be formed from each clique 
     *  by combining it with other cliques in its sublist. Used for finding 
     *  the amount of memory to allocate for new clique list node.  */
    long long unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (currCliques.numVertices - 1)) return;

    unsigned int sublist = currCliques.sublistIDs[idx];
    unsigned int vertex = currCliques.vertexIDs[idx];
    //get the neighbor list start & length for src
    int currentStart = graph.CsrT::GetNeighborListOffset(vertex);
    int listEnd = graph.CsrT::GetNeighborListLength(vertex) + currentStart - 1;
    //need to search within that list for dest (binary search)
    long long unsigned int numNew = 0llu;
    long long unsigned int j = idx + 1;
    while((j < currCliques.numVertices) && (currCliques.sublistIDs[j] == sublist)) {
        int destIndex = gunrock::util::BinarySearch(currCliques.vertexIDs[j], graph.column_indices, currentStart, listEnd);
        if ((destIndex <= listEnd) && (graph.column_indices[destIndex] == currCliques.vertexIDs[j])) {
            numNew++;
        }
        j++;
    }

    if ((numNew + currCliques.k) < w) {
        numNew = 0llu;
    }
    counts[idx] = numNew;
}

__global__ void mergeCliques(struct clique_node newCliques, struct clique_node currCliques, CsrT graph, long long unsigned int* offsets)
{
    /*  Merges k-cliques from newCliques.previous with any matching 
     *  k-cliques in its sublist into k+1-cliques and stores them in newCliques.
     *  (newCliques should have an appropriately sized empty node as head)
     *  offsets = number of new cliques formed in preceding cliques 
     *  Sizes of new sublists, used to compute offsets, are found by the countCliques kernel.   */
    long long unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (currCliques.numVertices - 1)) return;

    long long unsigned int cliqueOffset = offsets[idx];
    if (cliqueOffset == offsets[idx + 1]) return;   //if count=0
    unsigned int sublist = currCliques.sublistIDs[idx];
    unsigned int vertex = currCliques.vertexIDs[idx];
    //get the neighbor list start & length for src
    int currentStart = graph.CsrT::GetNeighborListOffset(vertex);
    int listEnd = graph.CsrT::GetNeighborListLength(vertex) + currentStart - 1;
    //need to search within that list for dest (binary search)
    long long unsigned int j = idx + 1;
    long long unsigned int count = 0;
    while((j < currCliques.numVertices) && (currCliques.sublistIDs[j] == sublist)) {
        int destIndex = gunrock::util::BinarySearch(currCliques.vertexIDs[j], graph.column_indices, currentStart, listEnd);
        if ((destIndex <= listEnd) && (graph.column_indices[destIndex] == currCliques.vertexIDs[j])) {
            newCliques.vertexIDs[cliqueOffset + count] = currCliques.vertexIDs[j];
            newCliques.sublistIDs[cliqueOffset + count] = idx;
            count++;
        }
        j++;
    }
}

__global__ void getVertexDegrees(CsrT graph, unsigned int* vertexLabels, unsigned int* vertexDegrees)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= graph.nodes) return;
   
    vertexLabels[idx] = idx;
    int numNeighbors = graph.CsrT::GetNeighborListLength(idx);
    vertexDegrees[idx] = numNeighbors;
}

__host__ unsigned int greedyHeuristic(CsrT graph, unsigned int* d_vertexList, unsigned int numVertices)
{
    /*  Basic greedy heuristic to find a large clique to use for the initial lower bound
     *  for pruning. This is the "single-run heuristic". In each iteration, the vertex with
     *  the largest degree or core number is chosen to be added to the heuristic clique, and
     *  the vertex list is filtered to remove vertices not connected to this new vertex.
     *  The filtering operation is performed in parallel on the GPU with the CUB Select operation.
     *  Input vertex list must be pre-sorted by either k-core or degree.    */
    cudaError_t retval = cudaSuccess;

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    unsigned int cliqueSize = 0;
    unsigned int* d_vertexList_temp;
    cudaMalloc((void**) &d_vertexList_temp, numVertices * sizeof(unsigned int));
    cudaMemcpy(d_vertexList_temp, d_vertexList, numVertices * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    unsigned int h_newVertex;
    cudaMemcpy(&h_newVertex, d_vertexList, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //Filter out vertices not connected to first vertex
    unsigned int* d_numRemaining;
    cudaMalloc((void**) &d_numRemaining, sizeof(unsigned int));
    unsigned int* d_filteredVertices;
    cudaMalloc((void**) &d_filteredVertices, numVertices * sizeof(unsigned int));
    unsigned int h_numRemaining = numVertices;

    while(h_numRemaining > 0) {
        cliqueSize++;
        IsNeighbor select_op(graph, h_newVertex);
        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        GUARD_CU(CUDAErrorCheck());
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_vertexList_temp, d_filteredVertices, d_numRemaining, h_numRemaining, select_op);
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_vertexList_temp, d_filteredVertices, d_numRemaining, h_numRemaining, select_op);
        GUARD_CU(CUDAErrorCheck());
    
        cudaMemcpy(&h_numRemaining, d_numRemaining, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        //Select next vertex from candidate list
        cudaMemcpy(&h_newVertex, d_filteredVertices, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        unsigned int* d_pointerHolder = d_filteredVertices;
        d_filteredVertices = d_vertexList_temp;
        d_vertexList_temp = d_pointerHolder;
    }

    cudaFree(d_vertexList_temp);
    cudaFree(d_numRemaining);
    cudaFree(d_filteredVertices);

    return cliqueSize;
}


__global__ void getNeighborCounts(CsrT graph, unsigned int* vertexList, unsigned int* neighborCount, unsigned int numVertices)
{
    /*  First set up step in parallel heuristic:
     *  Get number of neighbors for each vertex to allocate memory. */
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;
   
    unsigned int vertex = vertexList[idx];
    int numNeighbors = graph.CsrT::GetNeighborListLength(vertex);
    neighborCount[idx] = numNeighbors;
}

__global__ void setUpNeighborsThresholds(CsrT graph, unsigned int* vertexList, unsigned int* neighborCounts, unsigned int* thresholds_in, unsigned int* indices, unsigned int numVertices, unsigned int* neighbors, unsigned int* neighborThresholds)
{
    /*  Second set up step in parallel heuristic:
     *  Gather threshold values for all neighbors to enable the selection
     *  of the neighbor with highest threshold value to add to the heuristic 
     *  clique in each iteration of the main loop of the heuristic.    */
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;

    unsigned int vertex = vertexList[idx];
    unsigned int n = neighborCounts[idx];
    unsigned int offset = indices[idx];
    int eStart = graph.CsrT::GetNeighborListOffset(vertex);
    int eEnd = eStart + n;
    int count = 0;
    for (int e = eStart; e < eEnd; e++) {
        int dst = graph.CsrT::GetEdgeDest(e);
        neighbors[offset + count] = dst;
        neighborThresholds[offset + count] = thresholds_in[dst];
        count++;
    }
}

__global__ void checkConnections(CsrT graph, unsigned int* candidates, unsigned int* indices, cub::KeyValuePair<int, unsigned int>* bestVertices, unsigned int numSegments, unsigned int numCandidates, char* keepFlags, unsigned int* connectedCounts)
{
    /*  Step in main loop of parallel heuristic:
     *  Uses the vertices identified as having the highest threshold value 
     *  (k-core or degree) as the newly-added vertex for each clique. 
     *  Each thread checks other vertices in segment to see if they are connected
     *  to the newly-added vertex. Outputs flags for which vertices to keep and
     *  the total count for vertices remaining in segment.     */
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSegments) return;

    unsigned int startIndex = indices[idx];
    unsigned int nextStart = indices[idx + 1];
    if (bestVertices[idx].value == 0) {
        connectedCounts[idx] = 0;
        return;
    }
    unsigned int newVertex = candidates[bestVertices[idx].key + startIndex];
    //get the neighbor list start & length for src
    int currentStart = graph.CsrT::GetNeighborListOffset(newVertex);
    int listEnd = graph.CsrT::GetNeighborListLength(newVertex) + currentStart - 1;
    unsigned int numConnected = 0;
    //check all vertices in segment to see if connected to chosen vertex
    for (int u = startIndex; u < nextStart; u++) {
        if(u >= numCandidates) {
            printf("vertex index (%u) greater than number of candidates (%u)\n", u, numCandidates);
            return;
        }
        unsigned int currentVertex = candidates[u];
        if (currentVertex == newVertex) {
            keepFlags[u] = 0;
            continue; 
        }
        int destIndex = gunrock::util::BinarySearch(currentVertex, graph.column_indices, currentStart, listEnd);
        if (graph.column_indices[destIndex] == currentVertex) {
            keepFlags[u] = 1;
            numConnected++;
            currentStart = destIndex;
        }
        else {
            keepFlags[u] = 0;
        }
    }

    connectedCounts[idx] = numConnected;
}

__host__ unsigned int parallelGreedyHeuristic(CsrT graph, unsigned int* d_vertexList, unsigned int* d_thresholds_in, unsigned int numVertices)
{
    cudaError_t retval = cudaSuccess;

    //Setup:
    unsigned int* d_neighborCounts;
    cudaMalloc((void**) &d_neighborCounts, numVertices * sizeof(unsigned int));
    cudaMemset(d_neighborCounts, 0, numVertices * sizeof(unsigned int));
    //Kernel: one vertex per thread; return number of neighbors
    GUARD_CU(CUDAErrorCheck());
    getNeighborCounts<<<(numVertices + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(graph, d_vertexList, d_neighborCounts, numVertices);
    GUARD_CU(CUDAErrorCheck());

    //Scan to find total number of neighbors and start indices for each thread's group
    unsigned int* d_indices;
    cudaMalloc((void**) &d_indices, (numVertices + 1) * sizeof(unsigned int));
    cudaMemset(d_indices, 0, (numVertices + 1) * sizeof(unsigned int));
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_neighborCounts, d_indices, numVertices));
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_neighborCounts, d_indices, numVertices));

    //Allocate memory for neighbor lists and their threshold values
    unsigned int numCandidates_nextLast;
    cudaMemcpy(&numCandidates_nextLast, d_indices + (numVertices - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int numNeighborsLast;
    cudaMemcpy(&numNeighborsLast, d_neighborCounts + (numVertices - 1), sizeof(unsigned int), cudaMemcpyDeviceToHost);
    unsigned int numCandidates = numCandidates_nextLast + numNeighborsLast;
    //printf("numCandidates: %u\n", numCandidates);
    cudaMemcpy((d_indices + numVertices), &numCandidates, sizeof(unsigned int), cudaMemcpyHostToDevice);
    unsigned int* d_candidates;
    cudaMalloc((void**) &d_candidates, numCandidates * sizeof(unsigned int));
    cudaMemset(d_candidates, 0, numCandidates * sizeof(unsigned int));
    unsigned int* d_neighborThresholds;
    cudaMalloc((void**) &d_neighborThresholds, numCandidates * sizeof(unsigned int));
    cudaMemset(d_neighborThresholds, 0, numCandidates * sizeof(unsigned int));

    //Kernel: one vertex per thread; return neighbors and their thresholds
    GUARD_CU(CUDAErrorCheck());
    setUpNeighborsThresholds<<<(numVertices + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(graph, d_vertexList, d_neighborCounts, d_thresholds_in, d_indices, numVertices, d_candidates, d_neighborThresholds);
    GUARD_CU(CUDAErrorCheck());

    cudaFree(d_neighborCounts);

    unsigned int numSegments = numVertices;
    unsigned int cliqueSize = 1;
    cub::KeyValuePair<int, unsigned int>* d_maxIndices;
    cudaMalloc((void**) &d_maxIndices, numSegments * sizeof(cub::KeyValuePair<int, unsigned int>));
    char* d_keepFlags;
    cudaMalloc((void**) &d_keepFlags, numCandidates * sizeof(char));
    unsigned int* d_connectedCounts;
    cudaMalloc((void**) &d_connectedCounts, numSegments * sizeof(unsigned int));
    unsigned int* d_candidates_out;
    cudaMalloc((void**) &d_candidates_out, numCandidates * sizeof(unsigned int));
    unsigned int* d_neighborThresholds_out;
    cudaMalloc((void**) &d_neighborThresholds_out, numCandidates * sizeof(unsigned int));
    unsigned int* d_numCandidates_out;
    cudaMalloc((void**) &d_numCandidates_out, sizeof(unsigned int));
    unsigned int* d_nonzeroCounts;
    cudaMalloc((void**) &d_nonzeroCounts, numSegments * sizeof(unsigned int));
    unsigned int* d_numSegments_out;
    cudaMalloc((void**) &d_numSegments_out, sizeof(unsigned int));

    while(numSegments > 0) {
        //Segmented max reduce over threshold values to find next vertex to add in each segment
        cudaMemset(d_maxIndices, 0, numSegments * sizeof(cub::KeyValuePair<int, unsigned int>));
        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        CubDebugExit(cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_neighborThresholds, d_maxIndices, numSegments, d_indices, d_indices + 1));
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        CubDebugExit(cub::DeviceSegmentedReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_neighborThresholds, d_maxIndices, numSegments, d_indices, d_indices + 1));

        //Kernel: check if vertices are connected to new vertex, flag whether to keep, output number of vertices remaining
        cudaMemset(d_keepFlags, 0, numCandidates * sizeof(char));
        cudaMemset(d_connectedCounts, 0, numSegments * sizeof(unsigned int));
        GUARD_CU(CUDAErrorCheck());
        checkConnections<<<(numSegments + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(graph, d_candidates, d_indices, d_maxIndices, numSegments, numCandidates, d_keepFlags, d_connectedCounts);
        GUARD_CU(CUDAErrorCheck());

        //Keep only flagged for both candidate lists and threshold lists
        cudaMemset(d_candidates_out, 0, numCandidates * sizeof(unsigned int));
        cudaMemset(d_neighborThresholds_out, 0, numCandidates * sizeof(unsigned int));
        cudaMemset(d_numCandidates_out, 0, sizeof(unsigned int));
        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_candidates, d_keepFlags, d_candidates_out, d_numCandidates_out, numCandidates));
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_candidates, d_keepFlags, d_candidates_out, d_numCandidates_out, numCandidates));
        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_neighborThresholds, d_keepFlags, d_neighborThresholds_out, d_numCandidates_out, numCandidates));
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        CubDebugExit(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_neighborThresholds, d_keepFlags, d_neighborThresholds_out, d_numCandidates_out, numCandidates));
        cudaMemcpy(&numCandidates, d_numCandidates_out, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cliqueSize++;
        if(numCandidates == 0) break;

        //Compact out segments with count = 0
        cudaMemset(d_nonzeroCounts, 0, numSegments * sizeof(unsigned int));
        cudaMemset(d_numSegments_out, 0, sizeof(unsigned int));
        GreaterThan select_op(0);
        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        CubDebugExit(cub::DeviceSelect::If (d_temp_storage, temp_storage_bytes, d_connectedCounts, d_nonzeroCounts, d_numSegments_out, numSegments, select_op));
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        CubDebugExit(cub::DeviceSelect::If (d_temp_storage, temp_storage_bytes, d_connectedCounts, d_nonzeroCounts, d_numSegments_out, numSegments, select_op));
        cudaMemcpy(&numSegments, d_numSegments_out, sizeof(unsigned int), cudaMemcpyDeviceToHost);

        //Scan on number of vertices remaining to get new indices for segments
        cudaMemset(d_indices, 0, (numSegments + 1) * sizeof(unsigned int));
        d_temp_storage = NULL;
        temp_storage_bytes = 0;
        CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_nonzeroCounts, d_indices, (numSegments + 1)));
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_nonzeroCounts, d_indices, (numSegments + 1)));

        //Swap pointers
        unsigned int* temp = d_neighborThresholds;
        d_neighborThresholds = d_neighborThresholds_out;
        d_neighborThresholds_out = temp;
        temp = d_candidates;
        d_candidates = d_candidates_out;
        d_candidates_out = temp;
    }

    cudaFree(d_indices);
    cudaFree(d_candidates);
    cudaFree(d_neighborThresholds);
    cudaFree(d_maxIndices);
    cudaFree(d_keepFlags);
    cudaFree(d_connectedCounts);
    cudaFree(d_candidates_out);
    cudaFree(d_neighborThresholds_out);
    cudaFree(d_numCandidates_out);
    cudaFree(d_nonzeroCounts);
    cudaFree(d_numSegments_out);

    return cliqueSize;
}

__host__ cudaError_t findMaxCliquesGPU(std::string test_name, int argc, char** argv, struct clique_node** outputCliques)
{
    cudaError_t retval = cudaSuccess;

    //Define command line parameters:
    gunrock::util::Parameters parameters("test max clique");
    parameters.Use<int>("device", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, 0, "GPU device indices used for testing", __FILE__, __LINE__);
    parameters.Use<std::string>("timing", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, "all", "which parts to print timing for (none, total, preproc, loop, windowed, bfs, all)", __FILE__, __LINE__);
    parameters.Use<unsigned int>("num_runs", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, 1, "number of times to run maximum clique and average the runtimes (for collecting overall performance numbers)", __FILE__, __LINE__);
    parameters.Use<bool>("overall_perf", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, true, "if false, allows collection of additional metrics that may increase overall runtime", __FILE__, __LINE__);
    parameters.Use<bool>("num_cliques", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, false, "print number of cliques in each iteration", __FILE__, __LINE__);
    parameters.Use<std::string>("json_label", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, "" , "label to add to json filename to ensure no matching filenames", __FILE__, __LINE__);
    parameters.Use<unsigned int>("expected_max", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, 0, "expected maximum clique size for validation", __FILE__, __LINE__);
    parameters.Use<unsigned int>("lowerbound", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, 0, "input a known lower bound on the maximum clique size to improve pruning", __FILE__, __LINE__);
    parameters.Use<std::string>("heuristic", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, "multi_greedy" , "what kind of initial heuristic to use to find initial lower bound for pruning (none, greedy, multi_greedy)", __FILE__, __LINE__);
    parameters.Use<float>("frac_seeds", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, 1.0, "fraction of vertices (in range 0.0 to 1.0) to use as seeds for the multi-run heuristic", __FILE__, __LINE__);
    parameters.Use<std::string>("pruning", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, "simple" , "what kind of pruning to perform between iterations of the main loop (none, simple)", __FILE__, __LINE__);
    parameters.Use<bool>("kcore", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, false, "compute core numbers for all vertices to use in heuristic and pruning", __FILE__, __LINE__);
    parameters.Use<bool>("bfs", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, true, "solve max clique using full breadth-first exploration of search space", __FILE__, __LINE__);
    parameters.Use<bool>("windowing", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, false, "fully solve max clique for one subset of sublists at a time, rather than a fully breadth-first exploration of search space", __FILE__, __LINE__);
    parameters.Use<unsigned int>("window_size", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, 8192, "size of clique list windows for windowed version", __FILE__, __LINE__);
    parameters.Use<bool>("sort_sublists", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, true, "sort source vertices for 2-clique sublists based on their degree/core number (sublist_descend parameter determines whether this is ascending or descending order). only relevant when using windowing.", __FILE__, __LINE__);
    parameters.Use<bool>("sublist_descend", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, true, "whether to sort sublists (the source vertices) in descending (if true) or ascending (if false) degree/core number order", __FILE__, __LINE__);
    parameters.Use<std::string>("orientation", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, "degree" , "how to decide which edge to keep from each pair in the undirected graph (index, degree)", __FILE__, __LINE__);
    parameters.Use<bool>("order_candidates", REQUIRED_ARGUMENT | SINGLE_VALUE | OPTIONAL_PARAMETER, true , "sort the candidate vertices within the sublists (the dest vertices) from low degree to high degree", __FILE__, __LINE__);

    //Define internal parameters:
    parameters.Use<float>("load-time", REQUIRED_ARGUMENT | SINGLE_VALUE | INTERNAL_PARAMETER, 0, "time used to load / generate the graph", __FILE__, __LINE__);
    parameters.Use<int>("heuristic_clique_size", REQUIRED_ARGUMENT | SINGLE_VALUE | INTERNAL_PARAMETER, 0, "size of largest clique found by heuristic", __FILE__, __LINE__);
    parameters.Use<bool>("preempt_main", REQUIRED_ARGUMENT | SINGLE_VALUE | INTERNAL_PARAMETER, false, "flag to mark when preprocessing finds guaranteed maximum clique and main loop is preempted, or when pruning allows the main loop to finish early", __FILE__, __LINE__);
    parameters.Use<unsigned int>("kcore_max", REQUIRED_ARGUMENT | SINGLE_VALUE | INTERNAL_PARAMETER, 0, "size of largest kcore in graph", __FILE__, __LINE__);
    parameters.Use<unsigned int>("kcore_average", REQUIRED_ARGUMENT | SINGLE_VALUE | INTERNAL_PARAMETER, 0, "average vertex kcore decomposition value", __FILE__, __LINE__);
    parameters.Use<bool>("oom", REQUIRED_ARGUMENT | SINGLE_VALUE | INTERNAL_PARAMETER, false, "flag to mark when clique list is too large for GPU memory", __FILE__, __LINE__);
    parameters.Use<unsigned int>("peak_mem_use", REQUIRED_ARGUMENT | SINGLE_VALUE | INTERNAL_PARAMETER, 0, "size of largest clique list at any point in the computation (for windowed version)", __FILE__, __LINE__);
    parameters.Use<unsigned int>("num_seeds", REQUIRED_ARGUMENT | SINGLE_VALUE | INTERNAL_PARAMETER, 0, "number of vertices to use as seeds for multi-run heuristic, computed from frac_seeds input parameter", __FILE__, __LINE__);
    GUARD_CU(gunrock::graphio::UseParameters(parameters));
    GUARD_CU(gunrock::app::UseParameters_app(parameters));

    //Parse command line parameters:
    GUARD_CU(parameters.Parse_CommandLine(argc, argv));
    if (parameters.Get<bool>("help"))
    {
        parameters.Print_Help();
        return cudaSuccess;
    }
    parameters.Set("undirected", true);
    parameters.Set("sort-csr", true);
    int device = parameters.Get<int>("device");
    GUARD_CU(cudaSetDevice(device));
    bool perf = parameters.Get<bool>("overall_perf");
    bool quiet = parameters.Get<bool>("quiet");
    std::string timing = parameters.Get<std::string>("timing");
    bool output_num_cliques = parameters.Get<bool>("num_cliques");
    std::string heuristic = parameters.Get<std::string>("heuristic");
    std::string pruning = parameters.Get<std::string>("pruning");
    bool use_kcores = parameters.Get<bool>("kcore");
    bool sort_sublists = parameters.Get<bool>("sort_sublists");    //sort src vertices
    bool sublist_descend = parameters.Get<bool>("sublist_descend");
    std::string orientation = parameters.Get<std::string>("orientation");
    bool order_candidates = parameters.Get<bool>("order_candidates"); //sort dest vertices
    bool dfs = parameters.Get<bool>("windowing");
    bool bfs = parameters.Get<bool>("bfs");
    if ((heuristic == "none") && (pruning != "none")) {
        printf("ERROR: cannot prune intermediate results without heuristic\n");
        return cudaSuccess;
    }
    unsigned int num_runs = parameters.Get<unsigned int>("num_runs");
    if (!perf) {
        num_runs = 1;    //only do multiple runs to get average runtimes for collecting performance data
        parameters.Set("num_runs", 1);
    }
    GUARD_CU(parameters.Check_Required());

    //Read input graph
    gunrock::util::CpuTimer cpu_timer;
    GraphT graph;
    cpu_timer.Start();
    GUARD_CU(gunrock::graphio::LoadGraph(parameters, graph));
    cpu_timer.Stop();
    parameters.Set("load-time", cpu_timer.ElapsedMillis());

    //Move the graph to the GPU
    GUARD_CU(graph.Move(gunrock::util::HOST, gunrock::util::DEVICE));

    //Get GPU memory size
    size_t freeMem;
    size_t totalMem;
    GUARD_CU(cudaMemGetInfo (&freeMem, &totalMem));

    //Per-iteration (of clique merging loop) values: (only used if overall_perf=false)
    std::deque<float> loopRuntimes;

    //Per-run values (for averaging timing values over multiple runs)
    std::deque<struct time_breakdown> allRuntimes;

    //Timing values:
    struct dfs_loop_breakdown dfs_runtimes; //only for overall_perf=false
    struct bfs_loop_breakdown bfs_runtimes; //only for overall_perf=false
    cudaEvent_t start, beginHeuristic, endHeuristic, beginPresort, endPresort, beginTwoCliques, endTwoCliques, beginKCore, endKCore, beginPostSort, endPostSort, endPreproc, beginDFS, beginBFS, stop;
    cudaEvent_t startIteration, startFindWindow, endFindWindow, countingCliques, scanAlloc, mergingCliques;
    cudaEventCreate(&start);
    cudaEventCreate(&beginHeuristic);
    cudaEventCreate(&endHeuristic);
    cudaEventCreate(&beginPresort);
    cudaEventCreate(&endPresort);
    cudaEventCreate(&beginTwoCliques);
    cudaEventCreate(&endTwoCliques);
    cudaEventCreate(&beginKCore);
    cudaEventCreate(&endKCore);
    cudaEventCreate(&beginPostSort);
    cudaEventCreate(&endPostSort);
    cudaEventCreate(&endPreproc);
    cudaEventCreate(&beginDFS);
    cudaEventCreate(&beginBFS);
    cudaEventCreate(&startIteration);
    cudaEventCreate(&startFindWindow);
    cudaEventCreate(&endFindWindow);
    cudaEventCreate(&countingCliques);
    cudaEventCreate(&scanAlloc);
    cudaEventCreate(&mergingCliques);
    cudaEventCreate(&stop);

    unsigned int w = 2; //result of heuristic maximum clique 
    unsigned int numVertices = graph.nodes;
    struct clique_node* d_k_cliques;
    int k = 2;
    bool preemptMain = false;
    bool orientByDegree = false;
    struct clique_node* d_best_clique = new clique_node();
    unsigned int w_dfs = 2;
    std::deque<long long unsigned int> dfsCumulativeCliques((w_dfs - 1), 0);
    unsigned int peak_mem_use = 0;

    //Number of seeds for multi-run heuristic
    float frac_seeds = parameters.Get<float>("frac_seeds");
    if (frac_seeds > 1.0) {
        printf("ERROR: maximum value for frac_seeds is 1.0\n");
        return cudaSuccess;
    }
    unsigned int numSeeds = 0;
    if(heuristic == "multi_greedy") {
        if (frac_seeds == 1.0) {
            numSeeds = numVertices;
        }
        else {
            numSeeds = (((int)(numVertices * frac_seeds)) / 32) * 32;  //number of seeds rounded to warp size
            if (numSeeds == 0) numSeeds = 1;
        }
    }
    parameters.Set("num_seeds", numSeeds);
    if (!quiet) printf("number of seeds for heuristic: %u\n", numSeeds);

    for (unsigned int run = 0; run < num_runs; run++) {
        printf("starting run %u\n", run);
        GUARD_CU(cudaDeviceSynchronize());
        GUARD_CU(CUDAErrorCheck());
        cudaEventRecord(start);

        struct clique_node* d_two_cliques = *(outputCliques);
        d_two_cliques->numVertices = 0llu;
        d_two_cliques->k = 0;
        d_two_cliques->vertexIDs = NULL;
        d_two_cliques->sublistIDs = NULL;
        d_two_cliques->previous = NULL;
        w = 2;
        struct time_breakdown runtimes;
        unsigned int* d_degrees_in;
        cudaMalloc((void**) &d_degrees_in, numVertices * sizeof(unsigned int));
        cudaMemset(d_degrees_in, 0, sizeof(unsigned int));
        unsigned int* d_filterThresholds_in;
        cudaMalloc((void**) &d_filterThresholds_in, numVertices * sizeof(unsigned int));
        cudaMemset(d_filterThresholds_in, 0, sizeof(unsigned int));
        unsigned int* d_vertexLabels_in;
        cudaMalloc((void**) &d_vertexLabels_in, numVertices * sizeof(unsigned int));
        cudaMemset(d_vertexLabels_in, 0, sizeof(unsigned int));
        unsigned int* d_vertexLabels_out;
        cudaMalloc((void**) &d_vertexLabels_out, numVertices * sizeof(unsigned int));
        cudaMemset(d_vertexLabels_out, 0, sizeof(unsigned int));
        size_t temp_storage_bytes = 0;

        //Preprocessing:
        if (use_kcores) {   //compute k-core vertex decomposition using implementation from Gunrock
            cudaEventRecord(beginKCore);
            gunrock::util::Parameters kcore_parameters("use kcores in max clique");
            kcore_parameters = parameters;
            kcore_parameters.Set("quiet", true);
                
            GUARD_CU(gunrock::app::kcore::UseParameters(kcore_parameters));
            gunrock::util::Location target = gunrock::util::DEVICE;
            gunrock::app::kcore::Problem<GraphT> problem(kcore_parameters);
            gunrock::app::kcore::Enactor<gunrock::app::kcore::Problem<GraphT>> enactor;
            GUARD_CU(problem.Init(graph, target));
            GUARD_CU(enactor.Init(problem, target));
            GUARD_CU(problem.Reset(graph, target));
            GUARD_CU(enactor.Reset(target));
            GUARD_CU(enactor.Enact());

            //To get CPU array of k-core results:
            //unsigned int* h_k_cores = new unsigned int[graph.nodes];
            //GUARD_CU(problem.Extract(h_k_cores));
            /*printf("k-cores: ");
            for(int i = 0; i < graph.nodes; i++) {
                printf("%i, ", h_k_cores[i]);
            }
            printf("\n");*/
            //delete[] h_k_cores;

            auto &data_slice = problem.data_slices[0][0];
            cudaMemcpy(d_filterThresholds_in, data_slice.k_cores.GetPointer(gunrock::util::DEVICE), numVertices * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
            GUARD_CU(enactor.Release(target));
            GUARD_CU(problem.Release(target));
            cudaEventRecord(endKCore);

            GUARD_CU(CUDAErrorCheck());
            getVertexDegrees<<<(numVertices + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(graph, d_vertexLabels_in, d_degrees_in);
            GUARD_CU(CUDAErrorCheck());
        }   //end k-core computations

        else{   //get vertex degrees for heuristic
            GUARD_CU(CUDAErrorCheck());
            getVertexDegrees<<<(numVertices + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(graph, d_vertexLabels_in, d_filterThresholds_in);
            GUARD_CU(CUDAErrorCheck());
            d_degrees_in = d_filterThresholds_in;
        }

        if (sort_sublists || (heuristic != "none")) {
            unsigned int* d_filterThresholds_out;
            cudaMalloc((void**) &d_filterThresholds_out, numVertices * sizeof(unsigned int));

            //sort vertices by degree or core number
            if ((heuristic != "none") || sublist_descend) {
                cudaEventRecord(beginPresort);
                void* d_temp_storage = NULL;
                temp_storage_bytes = 0;
                cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_filterThresholds_in, d_filterThresholds_out, d_vertexLabels_in, d_vertexLabels_out, numVertices);
                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                cub::DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_filterThresholds_in, d_filterThresholds_out, d_vertexLabels_in, d_vertexLabels_out, numVertices);
                cudaFree(d_temp_storage);
                cudaEventRecord(endPresort);
            }

            if (heuristic == "greedy" || heuristic == "both") {
                //use a single run of the greedy heuristic
                cudaEventRecord(beginHeuristic);
                w = greedyHeuristic(graph, d_vertexLabels_out, numVertices);
                cudaEventRecord(endHeuristic);
                parameters.Set("heuristic_clique_size", w);
                if (!quiet) printf("initial greedy heuristic max clique size = %u\n", w);
            }

            if (heuristic == "multi_greedy" || heuristic == "both") {
                //run multiple instances of greedy heuristic in parallel with different seeds
                cudaEventRecord(beginHeuristic);
                w = parallelGreedyHeuristic(graph, d_vertexLabels_out, d_filterThresholds_in, numSeeds);
                cudaEventRecord(endHeuristic);
                parameters.Set("heuristic_clique_size", w);
                if (!quiet) printf("parallel greedy heuristic max clique size = %u\n", w);
            }
            
            if (sort_sublists && !sublist_descend) {
                //sort source vertices in ascending degree/core number order
                cudaEventRecord(beginPresort);
                void* d_temp_storage = NULL;
                temp_storage_bytes = 0;
                cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_filterThresholds_in, d_filterThresholds_out, d_vertexLabels_in, d_vertexLabels_out, numVertices);
                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_filterThresholds_in, d_filterThresholds_out, d_vertexLabels_in, d_vertexLabels_out, numVertices);
                cudaFree(d_temp_storage);
                cudaEventRecord(endPresort);
            }

            if (use_kcores && !perf) {   //k-core statistics
                //output maximum k-core value
                unsigned int h_maxKCore = 0u;
                if (sort_sublists && !sublist_descend) {
                    cudaMemcpy(&h_maxKCore, d_filterThresholds_out + numVertices - 1, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                }
                else {
                    cudaMemcpy(&h_maxKCore, d_filterThresholds_out, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                }
                if (!quiet) printf("maximum k-core = %u\n", h_maxKCore);
                parameters.Set("kcore_max", h_maxKCore);

                //find average k-core value
                unsigned int* d_kcoreSum;
                cudaMalloc((void**) &d_kcoreSum, sizeof(unsigned int));
                void* d_temp_storage = NULL;
                temp_storage_bytes = 0;
                CubDebugExit(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_filterThresholds_out, d_kcoreSum, numVertices));
                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                CubDebugExit(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_filterThresholds_out, d_kcoreSum, numVertices));
                cudaFree(d_temp_storage);
                unsigned int h_kcoreSum;
                cudaMemcpy(&h_kcoreSum, d_kcoreSum, sizeof(unsigned int), cudaMemcpyDeviceToHost);
                parameters.Set("kcore_average", (h_kcoreSum / numVertices));
                if (!quiet) printf("average k-core = %u\n", (h_kcoreSum / numVertices));
                cudaFree(d_kcoreSum);
            }

            cudaFree(d_filterThresholds_out);
        }   //end of heuristic and/or sorting source vertices

        unsigned int lowerbound_in = parameters.Get<unsigned int>("lowerbound");
        if (lowerbound_in > w) {
            w = lowerbound_in;
        }

        //Reset values (necessary for multiple runs)
        preemptMain = false;
        order_candidates = parameters.Get<bool>("order_candidates");
        orientByDegree = false;
        if (orientation == "degree") orientByDegree = true;
        //Set up first node of clique list data structure with 2-clique data (2-cliques are the edges)
        GUARD_CU(CUDAErrorCheck());
        if (sort_sublists == true) {
            cudaEventRecord(beginTwoCliques);
            setUpTwoCliques(graph, d_vertexLabels_out, d_degrees_in, d_filterThresholds_in, w, orientByDegree, order_candidates, d_two_cliques, beginPostSort, endPostSort, preemptMain);
            cudaEventRecord(endTwoCliques);
        }
        if (sort_sublists == false) {
            cudaEventRecord(beginTwoCliques);
            setUpTwoCliques(graph, d_vertexLabels_in, d_degrees_in, d_filterThresholds_in, w, orientByDegree, order_candidates, d_two_cliques, beginPostSort, endPostSort, preemptMain);
            cudaEventRecord(endTwoCliques);
        }
        GUARD_CU(CUDAErrorCheck());

        if (!quiet) printf("main loop %s preempted\n", preemptMain ? "is" : "is not");
        parameters.Set("preempt_main", preemptMain);

        cudaFree(d_vertexLabels_in);
        cudaFree(d_vertexLabels_out);
        cudaFree(d_filterThresholds_in);
        if (use_kcores) cudaFree(d_degrees_in);

        if (pruning != "simple") {
            w = 2;      //reset lower bound to measure effects of only pre-pruning vertices
        }

        d_k_cliques = d_two_cliques;
        if (!quiet) printf("number of 2-cliques: %llu\n", d_k_cliques->numVertices);
        //printSublists(d_k_cliques);

        cudaEventRecord(endPreproc);

        long long unsigned int sum = d_k_cliques->numVertices;
        if (output_num_cliques && !quiet) {
            printf("k\t size\t\t sum\n");
            printf("%i\t %llu\t\t %llu\n", d_k_cliques->k, d_k_cliques->numVertices, sum);
        }

        k = 2;
        unsigned int windowSize = parameters.Get<unsigned int>("window_size");   //max size of clique list to explore for windowed (semi-DFS) version
        unsigned int exploredTail = 0;  //end of fully-explored clique lists
        unsigned int startIndex = 0;

        d_best_clique->numVertices = 0llu;
        d_best_clique->k = 0;
        d_best_clique->vertexIDs = NULL;
        d_best_clique->sublistIDs = NULL;
        d_best_clique->previous = NULL;
        struct clique_node* d_window_cliques = new clique_node();
        int k_dfs = 2;
        w_dfs = w;
        long long unsigned int windowed_sum = 0ull;
        unsigned int window_mem_use = 0;

        //Run main algorithm: either full BFS, windowed version, or both (for comparison)
        cudaEventRecord(beginDFS);

        if (dfs && !preemptMain) {
            dfsCumulativeCliques.clear();
            dfsCumulativeCliques[0] = d_k_cliques->numVertices;
            if (!quiet) printf("run windowed maximum clique\n");
            while (exploredTail != (d_k_cliques->numVertices - 1)) { //run max clique algorithm for one subset of candidates at a time
                if (!perf) cudaEventRecord(startFindWindow);
                //find end of window such that it does not divide one sublist into two windows
                if (exploredTail != 0) startIndex = exploredTail + 1;
                if (parameters.Get<bool>("oom")) {
                    break;
                }

                unsigned int searchStartIndex = exploredTail + windowSize;
                long long unsigned int* h_endOfList = new long long unsigned int[0];
                h_endOfList[0] = d_k_cliques->numVertices - 1;
                long long unsigned int* d_tail;
                cudaMalloc((void**) &d_tail, sizeof(long long unsigned int));
                cudaMemcpy(d_tail, h_endOfList, sizeof(long long unsigned int), cudaMemcpyHostToDevice);
                if (searchStartIndex < (d_k_cliques->numVertices - 1)) {
                    GUARD_CU(CUDAErrorCheck());
                    findWindowTail<<<(windowSize + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(*d_k_cliques, windowSize, searchStartIndex, d_tail);
                    GUARD_CU(CUDAErrorCheck());
                }
                long long unsigned int h_tail;
                cudaMemcpy(&h_tail, d_tail, sizeof(long long unsigned int), cudaMemcpyDeviceToHost);

                long long unsigned int numCurrentCliques = h_tail - startIndex + 1;
                windowed_sum += numCurrentCliques;
                GUARD_CU(cudaMemGetInfo (&freeMem, &totalMem));
                size_t memNeeded = (1.01 * (2 * sizeof(long long unsigned int) * numCurrentCliques));
                if (freeMem <= memNeeded) {
                    parameters.Set("oom", true);
                    printf("clique list is too large to fit in GPU memory\n");
                    break;
                }
                
                d_window_cliques->numVertices = numCurrentCliques;
                d_window_cliques->k = d_k_cliques->k;
                d_window_cliques->vertexIDs = d_k_cliques->vertexIDs + startIndex;
                d_window_cliques->sublistIDs = d_k_cliques->sublistIDs + startIndex;
                d_window_cliques->previous = d_k_cliques->previous;
                k_dfs = d_k_cliques->k;

                if (!perf) {  //collect timings for loop sections
                    float elapsed;
                    cudaEventRecord(endFindWindow);
                    cudaEventSynchronize(endFindWindow);
                    cudaEventElapsedTime(&(elapsed), startFindWindow, endFindWindow);
                    dfs_runtimes.find_window += elapsed;
                    window_mem_use = 0;
                }

                while (d_window_cliques->numVertices > 1) { //find max cliques in this window
                    if (!perf) cudaEventRecord(startIteration);

                    //count number of new cliques generated
                    long long unsigned int* d_counts;
                    cudaMalloc((void**) &d_counts, numCurrentCliques * sizeof(long long unsigned int));

                    GUARD_CU(CUDAErrorCheck());
                    countNewCliques<<<(numCurrentCliques + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(*d_window_cliques, graph, w_dfs, d_counts);
                    GUARD_CU(CUDAErrorCheck());
                    if (!perf) cudaEventRecord(countingCliques);

                    //compute offsets with scan of counts array
                    long long unsigned int* d_offsets;
                    cudaMalloc((void**) &d_offsets, numCurrentCliques * sizeof(long long unsigned int));

                    void* d_temp_storage = NULL;  //reset these inside the loop
                    temp_storage_bytes = 0;
                    CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_counts, d_offsets, numCurrentCliques));
                    cudaMalloc(&d_temp_storage, temp_storage_bytes);
                    CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_counts, d_offsets, numCurrentCliques));
                    cudaFree(d_temp_storage);
                    
                    long long unsigned int numNewCliques;
                    cudaMemcpy(&numNewCliques, d_offsets + (numCurrentCliques - 1), sizeof(long long unsigned int), cudaMemcpyDeviceToHost);
                    if (numNewCliques == 0) break;

                    //Allocate memory for new linked list node
                    windowed_sum += numNewCliques;
                    long long unsigned int extraMem = (numNewCliques > numCurrentCliques) ? (numNewCliques - numCurrentCliques) : 0llu;
                    GUARD_CU(cudaMemGetInfo (&freeMem, &totalMem));
                    size_t memNeeded = (1.01 * ((2 * sizeof(unsigned int) * numNewCliques) + (2 * sizeof(long long unsigned int) * extraMem)));
                    if (freeMem <= memNeeded) {
                        parameters.Set("oom", true);
                        printf("clique list is too large to fit in GPU memory\n");
                        printf("free mem = %lu, new cliques space = %llu, extra counters space = %llu\n", freeMem, 2 * numNewCliques * sizeof(unsigned int), 2 * sizeof(long long unsigned int) * extraMem);
                        cudaFree(d_counts);
                        cudaFree(d_offsets);
                        break;
                    }
                    k_dfs++;
                    GUARD_CU(insertNewHeadNode(&d_window_cliques, numNewCliques, k_dfs));
                    if (!perf) cudaEventRecord(scanAlloc);

                    GUARD_CU(CUDAErrorCheck());
                    mergeCliques<<<(numCurrentCliques + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(*d_window_cliques, *(d_window_cliques->previous), graph, d_offsets);
                    GUARD_CU(CUDAErrorCheck());

                    if (output_num_cliques) {
                        printf("%i\t %llu\t\t %llu\n", d_window_cliques->k, d_window_cliques->numVertices, windowed_sum);
                    }

                    numCurrentCliques = numNewCliques;

                    if (!perf) {  //collect timings for loop sections
                        cudaEventRecord(mergingCliques);
                        float elapsed;
                        cudaEventSynchronize(mergingCliques);
                        cudaEventElapsedTime(&(elapsed), startIteration, countingCliques);
                        dfs_runtimes.count += elapsed;
                        cudaEventElapsedTime(&(elapsed), countingCliques, scanAlloc);
                        dfs_runtimes.scan_alloc += elapsed;
                        cudaEventElapsedTime(&(elapsed), scanAlloc, mergingCliques);
                        dfs_runtimes.merge += elapsed;
                        //update clique counts
                        if (k_dfs > w_dfs) {
                            dfsCumulativeCliques.push_back(numNewCliques);
                        }
                        else {
                            dfsCumulativeCliques[k_dfs - 2] += numNewCliques;
                        }
                        window_mem_use += numNewCliques;
                    }

                    GUARD_CU(cudaFree(d_counts));
                    GUARD_CU(cudaFree(d_offsets));
                }   //end while (exploration of window)
               
                if (!perf) {
                    if (window_mem_use > peak_mem_use) {
                        peak_mem_use = window_mem_use;
                    }
                }

                if (k_dfs >= w_dfs) {
                    if (!quiet) printf("new biggest clique size %u\n", k_dfs);

                    //free old best clique memory
                    while((d_best_clique->previous != NULL) && (d_best_clique->previous != d_k_cliques->previous)){//make sure not to free 2-cliques
                        GUARD_CU(cudaFree(d_best_clique->vertexIDs));
                        GUARD_CU(cudaFree(d_best_clique->sublistIDs));
                        d_best_clique = d_best_clique->previous;
                    }

                    //save clique list as best so far
                    cudaMemcpy(d_best_clique, d_window_cliques, sizeof(clique_node), cudaMemcpyHostToHost);
                    w_dfs = k_dfs;
                }
                //free memory if not best clique
                if ( (k_dfs < w_dfs) && (k_dfs != 2) ){
                    while ((d_window_cliques->previous != NULL) && (d_window_cliques->previous != d_k_cliques->previous)) {//make sure not to free 2-cliques
                        GUARD_CU(cudaFree(d_window_cliques->vertexIDs));
                        GUARD_CU(cudaFree(d_window_cliques->sublistIDs));
                        d_window_cliques = d_window_cliques->previous;
                    }
                }

                exploredTail = h_tail;
                if (parameters.Get<bool>("oom")) {
                    break;
                }
            }   //end while (all windows explored)
        }   //end dfs

        cudaEventRecord(beginBFS);
        
        if (bfs && !preemptMain) {
            if (!quiet) printf("run BFS max clique\n");
            //merge cliques
            //loop until no new cliques
            while(d_k_cliques->numVertices > 1) {

                if(!perf) cudaEventRecord(startIteration);

                //count number of new cliques generated
                long long unsigned int* d_counts;
                cudaMalloc((void**) &d_counts, d_k_cliques->numVertices * sizeof(long long unsigned int));
                long long unsigned int numCurrentCliques = d_k_cliques->numVertices;

                GUARD_CU(CUDAErrorCheck());
                countNewCliques<<<(numCurrentCliques + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(*d_k_cliques, graph, w, d_counts);
                GUARD_CU(CUDAErrorCheck());

                if(!perf) cudaEventRecord(countingCliques);

                //compute offsets with scan of counts array
                long long unsigned int* d_offsets;
                cudaMalloc((void**) &d_offsets, numCurrentCliques * sizeof(long long unsigned int));

                void* d_temp_storage = NULL;
                temp_storage_bytes = 0;
                CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_counts, d_offsets, numCurrentCliques));
                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_counts, d_offsets, numCurrentCliques));
                cudaFree(d_temp_storage);

                long long unsigned int numNewCliques;
                cudaMemcpy(&numNewCliques, d_offsets + (numCurrentCliques - 1), sizeof(long long unsigned int), cudaMemcpyDeviceToHost);
                if (numNewCliques == 0) break;
                
                //Allocate memory for new linked list node
                sum += numNewCliques;
                long long unsigned int extraMem = (numNewCliques > numCurrentCliques) ? (numNewCliques - numCurrentCliques) : 0llu;
                GUARD_CU(cudaMemGetInfo (&freeMem, &totalMem));
                size_t memNeeded = (1.01 * ((2 * sizeof(unsigned int) * numNewCliques) + (2 * sizeof(long long unsigned int) * extraMem)));
                if (freeMem <= memNeeded) {
                    parameters.Set("oom", true);
                    printf("clique list is larger than free memory\n");
                    printf("free mem = %lu, new cliques space = %llu, extra counters space = %llu\n", freeMem, 2 * numNewCliques * sizeof(unsigned int), 2 * sizeof(long long unsigned int) * extraMem);
                    GUARD_CU(cudaFree(d_counts));
                    GUARD_CU(cudaFree(d_offsets));
                    break;
                }
                k++;
                GUARD_CU(insertNewHeadNode(&d_k_cliques, numNewCliques, k));

                if(!perf) cudaEventRecord(scanAlloc);

                GUARD_CU(CUDAErrorCheck());
                mergeCliques<<<(numCurrentCliques + (BLOCK_SIZE - 1)) / BLOCK_SIZE, BLOCK_SIZE>>>(*d_k_cliques, *(d_k_cliques->previous), graph, d_offsets);
                GUARD_CU(CUDAErrorCheck());

                if(!perf){
                    cudaEventRecord(mergingCliques);
                    float elapsed;
                    cudaEventSynchronize(mergingCliques);
                    //Timings for parts of iteration:
                    cudaEventElapsedTime(&(elapsed), startIteration, countingCliques);
                    bfs_runtimes.count += elapsed;
                    cudaEventElapsedTime(&(elapsed), countingCliques, scanAlloc);
                    bfs_runtimes.scan_alloc += elapsed;
                    cudaEventElapsedTime(&(elapsed), scanAlloc, mergingCliques);
                    bfs_runtimes.merge += elapsed;
                    //Timing for entire iteration:
                    cudaEventElapsedTime(&(elapsed), startIteration, mergingCliques);
                    loopRuntimes.push_back(elapsed);
                }

                //printf("new cliques:\n");
                //printSublists(d_k_cliques);

                if (output_num_cliques) {
                    printf("%i\t %llu\t\t %llu\n", d_k_cliques->k, d_k_cliques->numVertices, sum);
                }

                cudaFree(d_counts);
                cudaFree(d_offsets);

                //Check if only the heuristic clique remains
                if ((numNewCliques + k - 1) == w) {
                    preemptMain = true;
                    parameters.Set("preempt_main", preemptMain);
                    printf("Heuristic clique is maximum. Maximum clique size = %u\n", w);
                    break;
                }
            }   //end while    
        }   //end if(bfs)
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        GUARD_CU(CUDAErrorCheck());

        if (!quiet) {

            if (!preemptMain) {
                if (dfs) {
                    printf("best clique(s) from DFS:\n");
                    printf("%llu %u-clique(s) found.\n", d_best_clique->numVertices, d_best_clique->k);
                }
                if (bfs) {
                    printf("best clique(s) from BFS:\n");
                    printf("%llu %u-clique(s) found.\n", d_k_cliques->numVertices, d_k_cliques->k);
                }
            }
        }

        cudaEventElapsedTime(&(runtimes.total), start, stop);
        if (heuristic != "none") cudaEventElapsedTime(&(runtimes.heuristic), beginHeuristic, endHeuristic);
        if (use_kcores) cudaEventElapsedTime(&(runtimes.kcore), beginKCore, endKCore);
        if (sort_sublists || (heuristic != "none")) cudaEventElapsedTime(&(runtimes.presort), beginPresort, endPresort);
        cudaEventElapsedTime(&(runtimes.two_cliques), beginTwoCliques, endTwoCliques);
        if (order_candidates) cudaEventElapsedTime(&(runtimes.postsort), beginPostSort, endPostSort);
        cudaEventElapsedTime(&(runtimes.total_preproc), start, endPreproc);
        if (dfs) cudaEventElapsedTime(&(runtimes.dfs), beginDFS, beginBFS);
        if (bfs) cudaEventElapsedTime(&(runtimes.bfs), beginBFS, stop);
        if ((timing == "all") || (timing == "total")) {
            printf("%f ms total\n", runtimes.total);
        }
        if ((timing == "all") || (timing == "heuristic")) {
            printf("%f ms for heuristic\n", runtimes.heuristic);
        }
        if  ((timing == "all") || (timing == "preproc")) {
            printf("%f ms for preprocessing\n", runtimes.total_preproc);
        }
        if ((timing == "all") || (timing == "loop")) {
            printf("%f ms in main loop(s)\n", (runtimes.dfs + runtimes.bfs));
        }
        if (((timing == "all") && (dfs == true)) || (timing == "windowed")) {
            printf("%f ms for windowed version\n", runtimes.dfs);
        }
        if (((timing == "all") && (bfs == true)) || (timing == "bfs")) {
            printf("%f ms for full bfs\n", runtimes.bfs);
        }

        allRuntimes.push_back(runtimes);
        if (run < (num_runs - 1)) {
            if (dfs) {
                while (d_best_clique->previous != NULL) {//make sure not to free 2-cliques
                    GUARD_CU(cudaFree(d_best_clique->vertexIDs));
                    GUARD_CU(cudaFree(d_best_clique->sublistIDs));
                    d_best_clique = d_best_clique->previous;
                }
            }
            if( bfs) {
                while ((d_k_cliques->previous != NULL)) {//make sure not to free 2-cliques
                    GUARD_CU(cudaFree(d_k_cliques->vertexIDs));
                    GUARD_CU(cudaFree(d_k_cliques->sublistIDs));
                    d_k_cliques = d_k_cliques->previous;
                }
                GUARD_CU(cudaFree(d_k_cliques->vertexIDs));
                GUARD_CU(cudaFree(d_k_cliques->sublistIDs));
            }
        }
    
        if (parameters.Get<bool>("oom")) {
            printf("skip rest of runs because this one was OOM\n");
            num_runs = run + 1;
            parameters.Set("num_runs", num_runs);
            break;
        }
        GUARD_CU(CUDAErrorCheck());
    }   //end all runs

    /*printf("sublists in all iterations:\n");
    struct clique_node* temp = d_k_cliques;
    while(temp != NULL){
        printf("%llu-cliques:\n", temp->k);
        printf("-------------------------------------\n");
        printSublists(temp);
        temp = temp->previous;
    }*/

    if (allRuntimes.size() != num_runs) {
        printf("ERROR: not all runs have saved timing info\n");
        return cudaSuccess;
    }
    struct time_breakdown avg_runtimes;
    for (struct time_breakdown i: allRuntimes) {
        avg_runtimes.total += i.total;
        avg_runtimes.heuristic += i.heuristic;
        avg_runtimes.kcore += i.kcore;
        avg_runtimes.presort += i.presort;
        avg_runtimes.two_cliques += i.two_cliques;
        avg_runtimes.postsort += i.postsort;
        avg_runtimes.total_preproc += i.total_preproc;
        avg_runtimes.dfs += i.dfs;
        avg_runtimes.bfs += i.bfs;
    }
    avg_runtimes.total /= num_runs;
    avg_runtimes.heuristic /= num_runs;
    avg_runtimes.kcore /= num_runs;
    avg_runtimes.presort /= num_runs;
    avg_runtimes.two_cliques /= num_runs;
    avg_runtimes.postsort /= num_runs;
    avg_runtimes.total_preproc /= num_runs;
    avg_runtimes.dfs /= num_runs;
    avg_runtimes.bfs /= num_runs; 

    //Output maximum clique(s)
    if (dfs && (d_best_clique->k > k)) {
        k = d_best_clique->k;
        *outputCliques = d_best_clique;
    }
    else{
        *outputCliques = d_k_cliques;
    }
    //Check if maximum clique matched expected size
    unsigned int expected = parameters.Get<unsigned int>("expected_max");
    unsigned int max_clique_size = k;
    if (preemptMain) max_clique_size = w;
    if (expected != 0) {
        if (max_clique_size == expected) printf("SUCCESS! Max clique size = %u\n", expected);
        else{
            printf("ERROR! Max clique size = %i. Expected value = %u\n", max_clique_size, expected);
        }
    }

    if (!quiet && !parameters.Get<bool>("oom")) {
        if (preemptMain) printKCliques_preempted(*outputCliques);
        else printKCliques(*outputCliques);
    }

    parameters.Set("peak_mem_use", peak_mem_use);
    if (!quiet) printf("peak memory use = %u\n", peak_mem_use);

    if (parameters.Get<bool>("json")) {
        outputJSON(test_name, graph, parameters, avg_runtimes, bfs_runtimes, dfs_runtimes, *outputCliques, loopRuntimes, dfsCumulativeCliques, allRuntimes);
    }

    //Free memory
    GUARD_CU(graph.Release());

    return retval;
}