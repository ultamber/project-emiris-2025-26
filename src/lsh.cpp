#include "lsh.hpp"
#include <random>
#include <limits>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

/**
 * Computes the L2 (Euclidean) distance between two vectors
 * @param a First vector
 * @param b Second vector
 * @return The Euclidean distance between vectors a and b
 */
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

/**
 * Builds the LSH index for the input dataset
 * @param data Input dataset to be indexed
 */
void LSH::buildIndex(const Dataset &data)
{
    data_ = data;
    dim_ = data.dimension;

    // Set default parameters if not provided
    w_ = args.w > 0 ? args.w : 4.0f;
    int L = args.L > 0 ? args.L : 10;
    int k = args.k > 0 ? args.k : 4;

    // TableSize heuristic ref 20): n/4
    tableSize_ = std::max<size_t>(1, data_.vectors.size() / 4);

    // Random number generators ref 18-19)
    std::mt19937_64 rng(args.seed);
    std::normal_distribution<double> normal(0.0, 1.0);                        // For random projections a
    std::uniform_real_distribution<float> unif(0.0f, w_);                     // For random shifts t
    std::uniform_int_distribution<uint32_t> distR(1u, (uint32_t)(MOD_M - 1)); // For hash combination r

    // Allocate hash function parameters
    a_.assign(L, std::vector<std::vector<float>>(k, std::vector<float>(dim_))); // Random projection vectors
    t_.assign(L, std::vector<float>(k, 0.0f));                                  // Random shifts
    r_.assign(L, std::vector<long long>(k));                                    // Integer coefficients for hash combination

    // Generate random projections, shifts, and integer coefficients ref 18
    for (int li = 0; li < L; ++li)
    {
        for (int j = 0; j < k; ++j)
        {
            // Generate random projection vector from standard normal
            for (int d = 0; d < dim_; ++d)
                a_[li][j][d] = (float)normal(rng);

            // Generate random shift uniformly in [0, w)
            t_[li][j] = unif(rng);

            // Generate random integer coefficient for hash combination ref 20
            r_[li][j] = (long long)distR(rng);
        }
    }

    // Allocate L hash tables with TableSize buckets each
    tables_.assign(L, std::vector<std::vector<std::pair<int, std::uint64_t>>>(tableSize_));

    // Insert all data points into hash tables
    for (int id = 0; id < (int)data_.vectors.size(); ++id)
    {
        const auto &p = data_.vectors[id].values;
        for (int li = 0; li < L; ++li)
        {
            // Compute ID(p) using hash function 
            // ref slides 18, 20-21
            std::uint64_t IDp = computeID(p, li);

            // Compute bucket index g(p) = ID(p) mod TableSize 
            // ref slide 20
            std::uint64_t g = IDp % tableSize_;

            // Store both point id and its ID for filtering 
            // ref slide 21
            tables_[li][g].push_back({id, IDp});
        }
    }

    if (std::getenv("LSH_DEBUG"))
    {
        std::cerr << "[LSH-DIAG] w=" << w_
                  << " L=" << L << " k=" << k
                  << " TableSize=" << tableSize_
                  << " min_h_seen=" << (min_h_seen_ == LLONG_MAX ? 0 : min_h_seen_)
                  << " neg_h_count=" << neg_h_count_
                  << "\n";
    }
}

/**
 * Computes the ID for a vector using LSH hash functions (slides 18, 20-21)
 * @param v Input vector
 * @param li Index of the hash table
 * @return 64-bit ID value
 */
std::uint64_t LSH::computeID(const std::vector<float> &v, int li) const
{
    std::uint64_t ID = 0;

    // Combine k hash functions 
    // ref slide 20
    for (int j = 0; j < args.k; ++j)
    {
        // Compute dot product for projection 
        // ref slide 18
        double dot = 0.0;
        for (int d = 0; d < dim_; ++d)
            dot += a_[li][j][d] * v[d];

        // Apply LSH hash function: h(p) = floor((a·p + t)/w)
        // slide 18
        long long hj = (long long)std::floor((dot + t_[li][j]) / w_);

        // Combine using random coefficients: ID = Σ r_ih_i(p) mod M 
        // slide 20
        ID = (ID + (std::uint64_t)(r_[li][j] * hj)) % MOD_M;

        // Update diagnostic counters
        if (hj < min_h_seen_)
            min_h_seen_ = hj;
        if (hj < 0)
            ++neg_h_count_;
    }

    return ID;
}

/**
 * Computes the hash key for a vector in a specific hash table
 * @param v Input vector to be hashed
 * @param li Index of the hash table
 * @return 64-bit hash key (bucket index)
 */
std::uint64_t LSH::keyFor(const std::vector<float> &v, int li) const
{
    std::uint64_t IDv = computeID(v, li);
    return IDv % tableSize_; // g(p) = ID(p) mod TableSize ref 20
}

/**
 * Performs LSH search for all queries
 * @param queries Query dataset
 * @param out Output file stream for results
 */
void LSH::search(const Dataset &queries, std::ofstream &out)
{
    using namespace std::chrono;
    out << "LSH\n\n";

    double totalAF = 0, totalRecall = 0, totalApprox = 0, totalTrue = 0;
    int qCount = (int)queries.vectors.size();

    // Optional query limit
    if (args.maxQueries > 0)
        qCount = std::min(qCount, args.maxQueries);

    for (int qi = 0; qi < qCount; ++qi)
    {
        const auto &q = queries.vectors[qi].values;
        auto t0 = high_resolution_clock::now();

        // multi probe lsh: collect candidates from multiple buckets
        std::vector<int> candidates;
        size_t examined = 0;

        // Hard cap on candidates to examine 
        // ref slides 13-14: stop after ~10L to 20L items
        size_t hardCap = args.rangeSearch ? 20 * args.L : 10 * args.L;

        // Probe all L tables
        for (int li = 0; li < args.L; ++li)
        {
            // compute query's hash value
            //  ref slide 20-21
            std::uint64_t IDq = computeID(q, li);
            std::uint64_t gq = IDq % tableSize_;

            // multi-probes: check main bucket + neighboring buckets (delta = -2 to +2)
            // significantly improves recall by finding near-collisions
            for (int delta = -2; delta <= 2; ++delta)
            {
                std::uint64_t gq2 = (gq + delta + tableSize_) % tableSize_;
                const auto &bucket = tables_[li][gq2];

                for (const auto &pr : bucket)
                {
                    if (delta == 0)
                    {
                        // Exact bucket: use ID filtering 
                        // ref slide 21 
                        if (pr.second == IDq)
                            candidates.push_back(pr.first);
                    }
                    else
                    {
                        // Neighboring buckets: accept all candidates 
                        candidates.push_back(pr.first);
                    }

                    if (++examined > hardCap)
                        break;
                }
                if (examined > hardCap)
                    break;
            }
            if (examined > hardCap)
                break;
        }

        // Deduplicate candidates 
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

        // Compute actual distances to candidates
        std::vector<std::pair<double, int>> distApprox;
        std::vector<int> rlist; // Range search results
        distApprox.reserve(candidates.size());

        for (int id : candidates)
        {
            double d = l2(q, data_.vectors[id].values);
            distApprox.emplace_back(d, id);

            // Range search: collect points within radius R 
            // ref slide 14
            if (args.rangeSearch && d <= args.R)
                rlist.push_back(id);
        }

        // Find top N nearest neighbors 
        // ref slide 13
        int N = std::min(args.N, (int)distApprox.size());
        if (N > 0)
        {
            std::nth_element(distApprox.begin(), distApprox.begin() + N, distApprox.end());
            std::sort(distApprox.begin(), distApprox.begin() + N);
        }

        double tApprox = duration<double>(high_resolution_clock::now() - t0).count();
        totalApprox += tApprox;

        // Compute true nearest neighbors for evaluation
        auto t2 = high_resolution_clock::now();
        std::vector<std::pair<double, int>> distTrue;
        distTrue.reserve(data_.vectors.size());
        for (auto &v : data_.vectors)
            distTrue.emplace_back(l2(q, v.values), v.id);
        std::nth_element(distTrue.begin(), distTrue.begin() + N, distTrue.end());
        std::sort(distTrue.begin(), distTrue.begin() + N);
        double tTrue = duration<double>(high_resolution_clock::now() - t2).count();
        totalTrue += tTrue;

        // Calculate quality metrics
        double AFq = 0, recallq = 0;
        for (int i = 0; i < N; ++i)
        {
            double da = distApprox[i].first, dt = distTrue[i].first;
            AFq += (dt > 0 ? da / dt : 1.0);

            // Check if approximate neighbor is in true top-N
            int aid = distApprox[i].second;
            for (int j = 0; j < N; ++j)
            {
                if (aid == distTrue[j].second)
                {
                    recallq += 1;
                    break;
                }
            }
        }

        if (N > 0)
        {
            AFq /= N;
            recallq /= N;
        }
        totalAF += AFq;
        totalRecall += recallq;

        // Output results
        out << "Query: " << qi << "\n"
            << std::fixed << std::setprecision(6);
        for (int i = 0; i < N; ++i)
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

    // Summary statistics
    out << "---- Summary (averages over queries) ----\n";
    out << "Average AF: " << (totalAF / qCount) << "\n";
    out << "Recall@N: " << (totalRecall / qCount) << "\n";
    out << "QPS: " << (1.0 / (totalApprox / qCount)) << "\n";
    out << "tApproximateAverage: " << (totalApprox / qCount) << "\n";
    out << "tTrueAverage: " << (totalTrue / qCount) << "\n";
}