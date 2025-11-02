#pragma once
#include <string>

struct Arguments {
    // File paths
    std::string inputFile;
    std::string queryFile;
    std::string outputFile;
    std::string type;

    // General
    int N = 1;
    double R = 1000.0;
    int seed = 1;
    bool rangeSearch = false;
    int maxQueries = 1000;

    // Algorithm selection
    bool useLSH = false;
    bool useHypercube = false;
    bool useIVFFlat = false;
    bool useIVFPQ = false;
    bool useBruteForce = false;

    // LSH
    int k = 4;
    int L = 5;
    double w = 4.0;

    // Hypercube
    int kproj = 8;
    int M = 200;
    int probes = 1000;
    int maxHamming = 0;

    // IVF / IVFPQ
    int kclusters = 64;
    int nprobe = 8;

    // PQ
    int nbits = 8;
    int Msubvectors = 16;
};
