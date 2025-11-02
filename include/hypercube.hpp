#pragma once
#include <unordered_map>
#include <vector>
#include "search_method.hpp"

class Hypercube : public SearchMethod{
private:
    Dataset data_;
    int dim_;
    int kproj_;  // d' in the PDF
    float w_;    // Bucket width for LSH
    
    // LSH hash function parameters ref 18, 24
    std::vector<std::vector<float>> proj_;   // Random projection vectors [kproj_][dim_]
    std::vector<float> shift_;                // Random shifts [kproj_]
    
    // Bit mapping functions f_j ref 24
    std::vector<std::uint64_t> fA_;  // Hash coefficients for f_j [kproj_]
    std::vector<std::uint64_t> fB_;  // Hash offsets for f_j [kproj_]
    
    // Hypercube storage ref 25
    bool denseCube_;  // Use dense or sparse representation
    std::vector<std::vector<int>> cubeDense_;  // Dense: [2^kproj_] vertices
    std::unordered_map<std::uint64_t, std::vector<int>> cubeSparse_;  // Sparse
    
    // Helper functions
    long long hij(const std::vector<float> &v, int j) const;
    bool fj(long long h, int j) const;
    std::uint64_t vertexOf(const std::vector<float> &v) const;
    std::vector<std::uint64_t> probesList(std::uint64_t base, int kproj, 
                                          int maxProbes, int maxHamming) const;
    double l2(const std::vector<float> &a, const std::vector<float> &b);
    
public:
    explicit Hypercube(const Arguments &a) : SearchMethod(a) {}
    void buildIndex(const Dataset &data);
    void search(const Dataset &queries, std::ofstream &out);
};