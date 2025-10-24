#include "lsh.hpp"
#include <random>
#include <limits>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

double LSH::l2(const std::vector<float> &a, const std::vector<float> &b)
{
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

void LSH::buildIndex(const Dataset &data)
{
    data_ = data;
    dim_ = data.dimension;
    w_ = args.w;
    int L = args.L, k = args.k;
    std::mt19937_64 rng(args.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, w_);

    a_.assign(L, std::vector<std::vector<float>>(k, std::vector<float>(dim_)));
    t_.assign(L, std::vector<float>(k, 0.0f));
    tables_.assign(L, {});

    for (int li = 0; li < L; ++li)
    {
        for (int j = 0; j < k; ++j)
        {
            for (int d = 0; d < dim_; ++d)
                a_[li][j][d] = (float)normal(rng);
            t_[li][j] = (float)unif(rng);
            t_min_ = std::min<double>(t_min_, t_[li][j]);
            t_max_ = std::max<double>(t_max_, t_[li][j]);
        }
    }

    for (int id = 0; id < (int)data_.vectors.size(); ++id)
    {
        for (int li = 0; li < L; ++li)
        {
            auto key = keyFor(data_.vectors[id].values, li);
            tables_[li][key].push_back(id);
        }
    }
    if (std::getenv("LSH_DEBUG")) {
    std::cerr << "[LSH-DIAG] w=" << w_
              << " t_range=[" << t_min_ << "," << t_max_ << "]"
              << " min_h_seen=" << (min_h_seen_==LLONG_MAX ? 0 : min_h_seen_)
              << " neg_h_count=" << neg_h_count_
              << "\n";
}
}

std::uint64_t LSH::keyFor(const std::vector<float> &v, int li) const
{
    std::uint64_t h = 1469598103934665603ULL; // FNV offset basis
    const std::uint64_t prime = 1099511628211ULL;

    for (int j = 0; j < args.k; ++j)
    {
        double dot = 0.0;
        for (int d = 0; d < dim_; ++d)
            dot += a_[li][j][d] * v[d];
        long long hj = (long long)std::floor((dot + t_[li][j]) / w_);
        h ^= (std::uint64_t)(hj * 11400714819323198485ull);
        h *= prime;
                
        // --- diagnostics ---
        if (hj < min_h_seen_) min_h_seen_ = hj;
        if (hj < 0) ++neg_h_count_;
    }
    return h;
}

void LSH::search(const Dataset &queries, std::ofstream &out)
{
    using namespace std::chrono;
    out << "LSH\n";

    double totalAF = 0.0, totalRecall = 0.0;
    double totalApproxTime = 0.0, totalTrueTime = 0.0;
    int queryCount = 0;

    auto startAll = high_resolution_clock::now();

    for (int qi = 0; qi < (int)queries.vectors.size(); ++qi)
    {
        const auto &q = queries.vectors[qi].values;
        auto startApprox = high_resolution_clock::now();

        // gather approximate candidates
        std::vector<int> cands;
        for (int li = 0; li < args.L; ++li)
        {
            auto key = keyFor(q, li);
            auto it = tables_[li].find(key);
            if (it != tables_[li].end())
                cands.insert(cands.end(), it->second.begin(), it->second.end());
        }

        double bestA = std::numeric_limits<double>::infinity();
        int idA = -1;
        std::vector<int> rlist;

        if (!cands.empty())
        {
            for (int id : cands)
            {
                double d = l2(q, data_.vectors[id].values);
                if (args.rangeSearch && d <= args.R)
                    rlist.push_back(id);
                if (d < bestA)
                {
                    bestA = d;
                    idA = id;
                }
            }
        }

        auto endApprox = high_resolution_clock::now();
        double tApprox = duration<double>(endApprox - startApprox).count();

        // compute true nearest
        auto startTrue = high_resolution_clock::now();
        double bestT = std::numeric_limits<double>::infinity();
        int idT = -1;
        for (int id = 0; id < (int)data_.vectors.size(); ++id)
        {
            double d = l2(q, data_.vectors[id].values);
            if (d < bestT)
            {
                bestT = d;
                idT = id;
            }
        }
        auto endTrue = high_resolution_clock::now();
        double tTrue = duration<double>(endTrue - startTrue).count();

        totalApproxTime += tApprox;
        totalTrueTime += tTrue;

        double AF = (bestT > 0.0 && bestA < std::numeric_limits<double>::infinity())
                        ? bestA / bestT
                        : std::numeric_limits<double>::infinity();
        double recall = (idA == idT) ? 1.0 : 0.0;

        if (idA != -1)
        {
            totalAF += AF;
            totalRecall += recall;
            queryCount++;
        }

        // per-query output
        out << "Query: " << qi << "\n";
        out << "Nearest neighbor-1: " << idA << "\n";
        out << "distanceApproximate: " << bestA << "\n";
        out << "distanceTrue: " << bestT << "\n";
        out << "R-near neighbors:\n";
        for (int id : rlist)
            out << id << "\n";
        out << "\n";
    }

    auto endAll = high_resolution_clock::now();
    double totalTime = duration<double>(endAll - startAll).count();

    double avgAF = (queryCount > 0) ? totalAF / queryCount : 0.0;
    double avgRecall = (queryCount > 0) ? totalRecall / queryCount : 0.0;
    double qps = (totalTime > 0) ? queryCount / totalTime : 0.0;
    double avgApprox = (queryCount > 0) ? totalApproxTime / queryCount : 0.0;
    double avgTrue = (queryCount > 0) ? totalTrueTime / queryCount : 0.0;

    out << std::fixed << std::setprecision(6);
    out << "Average AF: " << avgAF << "\n";
    out << "Recall@N: " << avgRecall << "\n";
    out << "QPS: " << qps << "\n";
    out << "tApproximateAverage: " << avgApprox << "\n";
    out << "tTrueAverage: " << avgTrue << "\n";
}
