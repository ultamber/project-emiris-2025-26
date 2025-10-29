#pragma once
#include "search_method.hpp"
#include <vector>

class IVFFlat : public SearchMethod
{
public:
    explicit IVFFlat(const Arguments &a) : SearchMethod(a) {}
    void buildIndex(const Dataset &data) override;
    void search(const Dataset &queries, std::ofstream &out) override;
    double silhouetteScore() const;
    std::vector<double> silhouettePerCluster() const;

private:
    Dataset data_;
    int dim_ = 0;
    int k_ = 0; // kclusters
    int nprobe_ = 0;

    std::vector<std::vector<float>> centroids_; // [k][dim]
    std::vector<std::vector<int>> lists_;       // [k] -> ids
    static double l2(const std::vector<float> &a, const std::vector<float> &b);
    void kmeans(const std::vector<std::vector<float>> &points, int k, int iters, unsigned seed);
};
