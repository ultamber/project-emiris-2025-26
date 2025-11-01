#pragma once
#include "search_method.hpp"
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cmath>

/**
 * Hypercube Approximate Nearest Neighbor search
 * - LSH-based hashing h_j(p) = floor((a_j · p + t_j) / w)
 * - Binary mapping f_j(h_j) ∈ {0,1} using a 2-universal hash
 * - Vertices g(p) = [f_1(h_1(p)), ..., f_{d'}(h_{d'}(p))] in {0,1}^{d'}
 * - Adaptive storage: dense for d' ≤ 24, sparse otherwise
 */
class Hypercube : public SearchMethod {
public:
    explicit Hypercube(const Arguments& a) : SearchMethod(a) {}
    void buildIndex(const Dataset& data) override;
    void search(const Dataset& queries, std::ofstream& out) override;

private:
    // === Dataset ===
    Dataset data_;
    int dim_ = 0;

    // === Parameters ===
    float w_ = 4.0f;                // bucket width, slides: w ∈ [2,6]
    int kproj_ = 0;                 // d' bits (Hypercube dimension)
    static constexpr std::uint64_t P32 = 4294967291ULL; // large 32-bit prime

    // === Base LSH (a_j, t_j) ===
    std::vector<std::vector<float>> proj_;  // random projection vectors a_j
    std::vector<float> shift_;               // random shifts t_j

    // === Bit mappers f_j(h_j) = ((A_j * h + B_j) mod P) & 1 ===
    std::vector<std::uint64_t> fA_, fB_;

    // === Hypercube storage ===
    bool denseCube_ = false;
    std::vector<std::vector<int>> cubeDense_;                      // for d' ≤ 24
    std::unordered_map<std::uint64_t, std::vector<int>> cubeSparse_; // for d' > 24

    // === Helpers ===
    static double l2(const std::vector<float>& a, const std::vector<float>& b);

    // Compute h_j(p) = floor((a_j · p + t_j) / w)
    inline long long hij(const std::vector<float>& v, int j) const {
        double dot = 0.0;
        for (int d = 0; d < dim_; ++d)
            dot += proj_[j][d] * v[d];
        return (long long)std::floor((dot + shift_[j]) / w_);
    }

    // Compute f_j(h_j) = ((A_j * h + B_j) mod P32) & 1
    inline uint8_t fj(long long h, int j) const {
        std::uint64_t hu = (std::uint64_t)((h % (long long)P32 + (long long)P32) % (long long)P32);
        std::uint64_t x  = (fA_[j] * hu + fB_[j]) % P32;
        return static_cast<uint8_t>(x & 1ULL);
    }

    // Compute vertex label g(p) = [f_1(h_1), ..., f_{d'}(h_{d'})]
    std::uint64_t vertexOf(const std::vector<float>& v) const;

    // Generate probing order in increasing Hamming distance
    std::vector<std::uint64_t> probesList(
        std::uint64_t base, int kproj, int maxProbes, int maxHamming) const;
};
