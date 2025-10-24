#pragma once
#include <vector>
#include <cstdint>
#include "search_method.hpp"

class IVFPQ : public SearchMethod {
public:
    explicit IVFPQ(const Arguments& a) : SearchMethod(a) {}
    void buildIndex(const Dataset& data) override;
    void search(const Dataset& queries, std::ofstream& out) override;

private:
    Dataset data_;
    int dim_=0;
    int k_=0;          // kclusters
    int nprobe_=0;     // coarse probes
    int M_=0;          // subquantizers
    int Ks_=256;       // 2^nbits
    std::vector<int> subdim_; // size M_

    // IVF
    std::vector<std::vector<float>> centroids_;    // [k][dim]
    std::vector<std::vector<int>> lists_;          // [k] -> ids

    // PQ (on residuals per coarse centroid, as taught)
    // Global PQ (minimal): train on all residuals together
    std::vector<std::vector<std::vector<float>>> codebooks_; // [M][Ks][subdim[m]]
    std::vector<std::vector<uint8_t>> codes_; // [N][M]

    // helpers
    static double l2(const std::vector<float>& a, const std::vector<float>& b);
    void kmeans(const std::vector<std::vector<float>>& P, int k, int iters, unsigned seed,
                std::vector<std::vector<float>>& C, std::vector<int>* assign=nullptr);
    void makeResiduals(const Dataset& data, const std::vector<int>& assign,
                       std::vector<std::vector<float>>& residuals);
    void trainPQ(const std::vector<std::vector<float>>& residuals);
    void encodeAll(const std::vector<std::vector<float>>& residuals);
};
