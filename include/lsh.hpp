#pragma once
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <climits>
#include "search_method.hpp"
class LSH : public SearchMethod
{
public:
    
    explicit LSH(const Arguments &a) : SearchMethod(a) {}
    void buildIndex(const Dataset &data) override;
    void search(const Dataset &queries, std::ofstream &out) override;

private:
    Dataset data_;
    int dim_ = 0;

    // L tables; each: key -> list of ids
    std::vector<std::unordered_map<std::uint64_t, std::vector<int>>> tables_;
    // L × k random projection vectors
    std::vector<std::vector<std::vector<float>>> a_; // [L][k][dim]
    // L × k random offsets t in [0,w)
    std::vector<std::vector<float>> t_; // [L][k]
    double w_ = 4.0;

    std::uint64_t keyFor(const std::vector<float> &v, int li) const;
    static double l2(const std::vector<float> &a, const std::vector<float> &b);

    // --- diagnostics (only active when LSH_DEBUG=1) ---
    mutable long long min_h_seen_ = LLONG_MAX;
    mutable size_t neg_h_count_ = 0;
    mutable double t_min_ = 1e300;
    mutable double t_max_ = -1e300;
};
