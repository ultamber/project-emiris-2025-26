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
    out << "LSH\n\n";

    double totalAF = 0.0, totalRecall = 0.0;
    double totalApproxTime = 0.0, totalTrueTime = 0.0;
    int queryCount = (int)queries.vectors.size();

    for (int qi = 0; qi < queryCount; ++qi)
    {
        const auto &q = queries.vectors[qi].values;

        // --- Approximate search ---
        auto t0 = high_resolution_clock::now();

        std::vector<int> candidates;
        for (int li = 0; li < args.L; ++li)
        {
            auto key = keyFor(q, li);
            auto it = tables_[li].find(key);
            if (it != tables_[li].end())
                candidates.insert(candidates.end(), it->second.begin(), it->second.end());
        }

        std::vector<std::pair<double, int>> distApprox;
        distApprox.reserve(candidates.size());
        std::vector<int> rlist;

        for (int id : candidates)
        {
            double d = l2(q, data_.vectors[id].values);
            distApprox.emplace_back(d, id);
            if (args.rangeSearch && d <= args.R)
                rlist.push_back(id);
        }

        int N = std::min(args.N, (int)distApprox.size());
        if (N > 0)
        {
            std::nth_element(distApprox.begin(), distApprox.begin() + N, distApprox.end());
            std::sort(distApprox.begin(), distApprox.begin() + N);
        }

        auto t1 = high_resolution_clock::now();
        double tApprox = duration<double>(t1 - t0).count();
        totalApproxTime += tApprox;

        // --- True nearest computation ---
        auto t2 = high_resolution_clock::now();
        std::vector<std::pair<double, int>> distTrue;
        distTrue.reserve(data_.vectors.size());
        for (const auto &v : data_.vectors)
            distTrue.emplace_back(l2(q, v.values), v.id);
        std::nth_element(distTrue.begin(), distTrue.begin() + N, distTrue.end());
        std::sort(distTrue.begin(), distTrue.begin() + N);
        auto t3 = high_resolution_clock::now();
        double tTrue = duration<double>(t3 - t2).count();
        totalTrueTime += tTrue;

        // --- Per-query metrics ---
        double AFq = 0.0;
        double recallq = 0.0;
        int found = 0;

        for (int ni = 0; ni < N; ++ni)
        {
            double da = distApprox[ni].first;
            double dt = distTrue[ni].first;
            AFq += (dt > 0.0 ? da / dt : 1.0);

            int approx_id = distApprox[ni].second;
            // check if this approx neighbor appears among true top-N
            for (int j = 0; j < N; ++j)
                if (approx_id == distTrue[j].second)
                {
                    recallq += 1.0;
                    break;
                }
            found++;
        }

        if (N > 0)
        {
            AFq /= N;
            recallq /= N;
        }

        totalAF += AFq;
        totalRecall += recallq;

        // --- Output formatting ---
        out << "Query: " << qi << "\n";
        out << std::fixed << std::setprecision(6);
        for (int ni = 0; ni < N; ++ni)
        {
            out << "Nearest neighbor-" << (ni + 1) << ": " << distApprox[ni].second << "\n";
            out << "distanceApproximate: " << distApprox[ni].first << "\n";
            out << "distanceTrue: " << distTrue[ni].first << "\n";
        }
        out << "\nR-near neighbors:\n";
        for (int id : rlist) out << id << "\n";
        out << "\n";
        out << "Average AF: " << AFq << "\n";
        out << "Recall@N: " << recallq << "\n";
        out << "QPS: " << (tApprox > 0.0 ? 1.0 / tApprox : 0.0) << "\n";
        out << "tApproximateAverage: " << tApprox << "\n";
        out << "tTrueAverage: " << tTrue << "\n\n";
    }

  // --- Summary over all queries ---
    double avgAF = totalAF / queryCount;
    double avgRecall = totalRecall / queryCount;
    double avgApprox = totalApproxTime / queryCount;
    double avgTrue = totalTrueTime / queryCount;
    double qps = (avgApprox > 0) ? 1.0 / avgApprox : 0.0;

    out << "---- Summary (averages over queries) ----\n";
    out << "Average AF: " << avgAF << "\n";
    out << "Recall@N: " << avgRecall << "\n";
    out << "QPS: " << qps << "\n";
    out << "tApproximateAverage: " << avgApprox << "\n";
    out << "tTrueAverage: " << avgTrue << "\n";
}