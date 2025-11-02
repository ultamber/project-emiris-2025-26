#pragma once
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <climits>
#include <cmath>
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
    const std::uint64_t MOD_M = 4294967291ULL; // 2^32 - 5, a large prime
    // --- LSH parameters ---
    size_t tableSize_ = 0;                                                        // TableSize = n / 4
    std::vector<std::vector<long long>> r_;                                       // random coefficients
    std::vector<std::vector<std::vector<std::pair<int, std::uint64_t>>>> tables_; // L x TableSize buckets
    // L × k random projection vectors
    std::vector<std::vector<std::vector<float>>> a_; // [L][k][dim]
    // L × k random offsets t in [0,w)
    std::vector<std::vector<float>> t_; // [L][k]
    double w_ = 4.0;
    // static constexpr std::uint64_t MOD_M = (1ull << 32) - 5;
    std::uint64_t keyFor(const std::vector<float> &v, int li) const;
    static double l2(const std::vector<float> &a, const std::vector<float> &b);

    // --- diagnostics (only active when LSH_DEBUG=1) ---
    mutable long long min_h_seen_ = LLONG_MAX;
    mutable size_t neg_h_count_ = 0;
    mutable double t_min_ = 1e300;
    mutable double t_max_ = -1e300;

    // Safe modular helpers
    inline static long long mod_ll(long long a, long long m)
    {
        long long r = a % m;
        return (r < 0) ? r + m : r;
    }
    inline static std::uint64_t mod_u64(std::uint64_t a, std::uint64_t m)
    {
        return (m == 0) ? 0 : (a % m);
    }

    // Compute base LSH function h_j(p)
    inline long long hij(const std::vector<float> &v, int li, int j) const
    {
        double dot = 0.0;
        for (int d = 0; d < dim_; ++d)
            dot += a_[li][j][d] * v[d];
        return (long long)std::floor((dot + t_[li][j]) / w_);
    }

    // Compute amplified ID(p) = Σ r_j h_j(p) mod M  ref 21
    std::uint64_t computeID(const std::vector<float> &v, int li) const;

};
