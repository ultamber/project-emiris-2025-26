#pragma once
#include "search_method.hpp"
#include <unordered_map>
#include <vector>
#include <cstdint>

class Hypercube : public SearchMethod {
public:
    explicit Hypercube(const Arguments& a) : SearchMethod(a) {}
    void buildIndex(const Dataset& data) override;
    void search(const Dataset& queries, std::ofstream& out) override;

private:
    Dataset data_;
    int dim_=0;

    // kproj random projection vectors
    std::vector<std::vector<float>> proj_; // [kproj][dim]
    // vertex -> ids
    std::unordered_map<std::uint64_t, std::vector<int>> cube_;

    std::uint64_t vertexOf(const std::vector<float>& v) const;
    static double l2(const std::vector<float>& a, const std::vector<float>& b);
    std::vector<std::uint64_t> probesList(std::uint64_t base, int kproj, int maxProbes) const;
};
