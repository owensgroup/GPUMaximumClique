//cliqueMerging.cuh
#ifndef CLIQUE_MERGING_CUH
#define CLIQUE_MERGING_CUH

#define BLOCK_SIZE 1024

#include <stdio.h>
#include <string>

enum {  //values for setting command line options
    NO_ARGUMENT        = 0x1,
    REQUIRED_ARGUMENT  = 0x2,
    OPTIONAL_ARGUMENT  = 0x4,

    SINGLE_VALUE       = 0x20,
    MULTI_VALUE        = 0x40,

    REQUIRED_PARAMETER = 0x100,
    OPTIONAL_PARAMETER = 0x200,
    INTERNAL_PARAMETER = 0x400,
};

struct clique_node {
    long long unsigned int numVertices;
    int k;
    unsigned int* vertexIDs;
    unsigned int* sublistIDs;
    struct clique_node* previous;
    clique_node();
    ~clique_node();
};

struct time_breakdown {
    float total;
    float heuristic;
    float kcore;
    float presort;
    float two_cliques;
    float postsort;
    float total_preproc;
    float dfs;
    float bfs;
    time_breakdown();
    ~time_breakdown();
};

struct bfs_loop_breakdown {
    float count;
    float scan_alloc;
    float merge;
    bfs_loop_breakdown();
    ~bfs_loop_breakdown();
};

struct dfs_loop_breakdown {
    float find_window;
    float count;
    float scan_alloc;
    float merge;
    dfs_loop_breakdown();
    ~dfs_loop_breakdown();
};

__host__ cudaError_t insertNewHeadNode(struct clique_node** oldHead, long long unsigned int size, int k);
    /*  Allocates memory for new node of given size in clique data structure.
     *  Sublist and vertex info will be empty.
     *  Modifies the head pointer to point to the new head node.    */

__host__ void printSublists(struct clique_node* currentNode);
    /*  Prints all the sublists stored in the current linked list node. */

__host__ unsigned int* readClique(struct clique_node* k_cliques, int cliqueSize, long long unsigned int index);
    /*  Returns the clique represented by the (index)th location in
     *  the node k_cliques. */

__host__ void printKCliques(struct clique_node* k_cliques);
    /*  Prints all the cliques (of size k) in node k_cliques.    */

__host__ cudaError_t findMaxCliquesGPU(std::string test_name, int argc, char** argv, struct clique_node** cliqueOutput);
    /*  Loads graph using the Gunrock graph loader. Format, filename,
     *  and other graph properties are input from the command line.
     *  Outputs the linked list of cliques, from which the maximum clique(s)
     *  can be read out. (using printKCliques() function)   */

//Experimental:
__host__ cudaError_t findKCliquesGPU(std::string test_name, int argc, char** argv);
    /*  Version of clique merging algorithm that does not retain
     *  the sublists from previous iterations in order to save memory
     *  and allow the algorithm to run for more iterations.     */

#endif
