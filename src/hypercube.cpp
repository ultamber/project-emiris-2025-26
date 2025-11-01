#include "hypercube.hpp"
#include <random>
#include <limits>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <chrono>
#include <iomanip>
#include <iostream>

/**
 * Computes the L2 (Euclidean) distance between two vectors
 * @param a First vector
 * @param b Second vector
 * @return The Euclidean distance between vectors a and b
 */
double Hypercube::l2(const std::vector<float> &a, const std::vector<float> &b)
{
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

/**
 * Builds the Hypercube index for the input dataset
 * @param data Input dataset to be indexed
 */
void Hypercube::buildIndex(const Dataset &data)
{
    data_ = data;
    dim_ = data.dimension;

    // Determine d' based on slides: floor(log2 n) - {1,2,3}
    const size_t n = data_.vectors.size();
    int dlog = (n > 0) ? (int)std::floor(std::log2((double)std::max<size_t>(1, n))) : 1;
    kproj_ = (args.kproj > 0) ? args.kproj : std::max(1, dlog - 2);

    w_ = (args.w > 0) ? args.w : 4.0f;

    std::mt19937_64 rng(args.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<float> unif(0.0f, w_);
    std::uniform_int_distribution<uint32_t> uni32(1u, 0xffffffffu);

    // Generate base LSH projections (a_j, t_j)
    proj_.assign(kproj_, std::vector<float>(dim_));
    shift_.assign(kproj_, 0.0f);
    for (int j = 0; j < kproj_; ++j)
    {
        for (int d = 0; d < dim_; ++d)
            proj_[j][d] = (float)normal(rng);
        shift_[j] = unif(rng);
    }

    // Generate bit-mapping functions f_j(h)
    fA_.resize(kproj_);
    fB_.resize(kproj_);
    for (int j = 0; j < kproj_; ++j)
    {
        fA_[j] = (std::uint64_t)uni32(rng);
        fB_[j] = (std::uint64_t)uni32(rng);
        if (fA_[j] == 0)
            fA_[j] = 1;
    }

    // Choose cube representation based on d'
    denseCube_ = (kproj_ <= 24);
    if (denseCube_)
    {
        cubeDense_.assign(1ULL << kproj_, {});
        cubeSparse_.clear();
    }
    else
    {
        cubeSparse_.clear();
        cubeDense_.clear();
    }

    // Insert all data points into their vertices
    for (int id = 0; id < (int)n; ++id)
    {
        const auto &v = data_.vectors[id].values;
        std::uint64_t vtx = vertexOf(v);
        if (denseCube_)
            cubeDense_[vtx].push_back(id);
        else
            cubeSparse_[vtx].push_back(id);
    }

    if (std::getenv("CUBE_DEBUG"))
    {
        std::cerr << "[CUBE] n=" << n
                  << " dim=" << dim_
                  << " d'=" << kproj_
                  << " w=" << w_
                  << " dense=" << denseCube_
                  << " vertices=" << (denseCube_ ? (1ULL << kproj_) : cubeSparse_.size())
                  << "\n";
    }
}

/**
 * Maps a vector to a vertex in the hypercube
 * @param v Input vector
 * @return 64-bit vertex label where each bit represents a projection result
 */
std::uint64_t Hypercube::vertexOf(const std::vector<float> &v) const
{
    std::uint64_t key = 0;
    for (int j = 0; j < kproj_; ++j)
    {
        long long hj = hij(v, j);
        if (fj(hj, j))
            key |= (1ULL << j);
    }
    return key;
}

/**
 * Generates a list of vertices to probe in order of increasing Hamming distance
 * @param base Starting vertex
 * @param kproj Number of projection dimensions (bits)
 * @param maxProbes Maximum number of vertices to return
 * @return Vector of vertex labels to probe
 */
std::vector<std::uint64_t>
Hypercube::probesList(std::uint64_t base, int kproj, int maxProbes, int maxHamming) const
{
    std::vector<std::uint64_t> out{base};
    if ((int)out.size() >= maxProbes)
        return out;

    const int Hmax = std::min(kproj, maxHamming);
    for (int h = 1; h <= Hmax && (int)out.size() < maxProbes; ++h)
    {
        std::vector<int> idx(h);
        std::iota(idx.begin(), idx.end(), 0);
        while (true)
        {
            std::uint64_t mask = 0;
            for (int i : idx)
                mask |= (1ULL << i);
            out.push_back(base ^ mask);
            if ((int)out.size() >= maxProbes)
                break;
            int i;
            for (i = h - 1; i >= 0 && idx[i] == i + kproj - h; --i)
                ;
            if (i < 0)
                break;
            ++idx[i];
            for (int j = i + 1; j < h; ++j)
                idx[j] = idx[j - 1] + 1;
        }
    }
    return out;
}

/**
 * Performs Hypercube search for all queries
 * @param queries Query dataset
 * @param out Output file stream for results
 */
void Hypercube::search(const Dataset &queries, std::ofstream &out)
{
    using namespace std::chrono;
    out << "Hypercube\n\n";

    int M = args.M;
    int probes = args.probes;
    int maxHam = (args.maxHamming > 0) ? args.maxHamming : kproj_;
    double R = args.R;
    bool doRange = args.rangeSearch;
    int N = args.N;

    double totalAF = 0, totalRecall = 0, totalApprox = 0, totalTrue = 0;
    int Q = (int)queries.vectors.size();

    for (int qi = 0; qi < Q; ++qi)
    {
        const auto &q = queries.vectors[qi].values;
        auto t0 = high_resolution_clock::now();

        std::unordered_set<int> cand;
        cand.reserve(std::min<size_t>((size_t)M, (size_t)4096));
        std::uint64_t base = vertexOf(q);
        auto plist = probesList(base, kproj_, probes, maxHam);

        size_t gathered = 0;
        for (auto vtx : plist)
        {
            const std::vector<int> *bucket = nullptr;
            if (denseCube_)
                bucket = &cubeDense_[vtx];
            else
            {
                auto it = cubeSparse_.find(vtx);
                if (it == cubeSparse_.end())
                    continue;
                bucket = &it->second;
            }
            for (int id : *bucket)
            {
                if (cand.insert(id).second && ++gathered >= (size_t)M)
                    break;
            }
            if (gathered >= (size_t)M)
                break;
        }

        // Compute distances for collected candidates
        std::vector<std::pair<double, int>> distApprox;
        distApprox.reserve(cand.size());
        std::vector<int> rlist;
        for (int id : cand)
        {
            double d = l2(q, data_.vectors[id].values);
            distApprox.emplace_back(d, id);
            if (doRange && d <= R)
                rlist.push_back(id);
        }

        int topN = std::min(N, (int)distApprox.size());
        if (topN > 0)
        {
            std::nth_element(distApprox.begin(), distApprox.begin() + topN, distApprox.end());
            std::sort(distApprox.begin(), distApprox.begin() + topN);
            distApprox.resize(topN);
        }
        double tApprox = duration<double>(high_resolution_clock::now() - t0).count();
        totalApprox += tApprox;

        // Ground truth
        auto t2 = high_resolution_clock::now();
        std::vector<std::pair<double, int>> distTrue;
        distTrue.reserve(data_.vectors.size());
        for (auto &v : data_.vectors)
            distTrue.emplace_back(l2(q, v.values), v.id);
        std::nth_element(distTrue.begin(), distTrue.begin() + topN, distTrue.end());
        std::sort(distTrue.begin(), distTrue.begin() + topN);
        distTrue.resize(topN);
        double tTrue = duration<double>(high_resolution_clock::now() - t2).count();
        totalTrue += tTrue;

        // Metrics
        double AFq = 0, recallq = 0;
        for (int i = 0; i < topN; ++i)
        {
            double da = distApprox[i].first, dt = distTrue[i].first;
            AFq += (dt > 0 ? da / dt : 1.0);
            int aid = distApprox[i].second;
            for (int j = 0; j < topN; ++j)
                if (aid == distTrue[j].second)
                {
                    recallq += 1;
                    break;
                }
        }
        if (topN > 0 && !distApprox.empty())
        {
            AFq /= distApprox.size();
            recallq /= topN;
        }
        totalAF += AFq;
        totalRecall += recallq;

        // Output per query
        out << "Query: " << qi << "\n"
            << std::fixed << std::setprecision(6);
        for (int i = 0; i < topN; ++i)
        {
            out << "Nearest neighbor-" << (i + 1) << ": " << distApprox[i].second << "\n";
            out << "distanceApproximate: " << distApprox[i].first << "\n";
            out << "distanceTrue: " << distTrue[i].first << "\n";
        }
        out << "\nR-near neighbors:\n";
        for (int id : rlist)
            out << id << "\n";
        out << "\n";
    }

    // Summary
    double avgAF = totalAF / Q, avgRecall = totalRecall / Q;
    double avgApprox = totalApprox / Q, avgTrue = totalTrue / Q;
    double qps = (avgApprox > 0) ? 1.0 / avgApprox : 0.0;
    out << "---- Summary (averages over queries) ----\n";
    out << "Average AF: " << avgAF << "\n";
    out << "Recall@N: " << avgRecall << "\n";
    out << "QPS: " << qps << "\n";
    out << "tApproximateAverage: " << avgApprox << "\n";
    out << "tTrueAverage: " << avgTrue << "\n";
}
