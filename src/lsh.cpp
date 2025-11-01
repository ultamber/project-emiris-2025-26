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
    // Store dataset and its dimension
    data_ = data;
    dim_ = data.dimension;
    w_ = args.w;                // Bucket width
    int L = args.L, k = args.k; // L hash tables, k hash functions per table

    // Initialize random number generators
    std::mt19937_64 rng(args.seed);
    std::normal_distribution<double> normal(0.0, 1.0);    // For random projections
    std::uniform_real_distribution<double> unif(0.0, w_); // For random shifts

    // Initialize hash function parameters
    a_.assign(L, std::vector<std::vector<float>>(k, std::vector<float>(dim_))); // Random projection vectors
    t_.assign(L, std::vector<float>(k, 0.0f));                                  // Random shifts
    tables_.assign(L, {});                                                      // Hash tables

    // Generate random projections and shifts for each hash function
    for (int li = 0; li < L; ++li)
    {
        for (int j = 0; j < k; ++j)
        {
            // Generate random projection vector
            for (int d = 0; d < dim_; ++d)
                a_[li][j][d] = (float)normal(rng);
            t_[li][j] = (float)unif(rng);
            t_min_ = std::min<double>(t_min_, t_[li][j]);
            t_max_ = std::max<double>(t_max_, t_[li][j]);
        }
    }

    // Insert all data points into hash tables
    for (int id = 0; id < (int)data_.vectors.size(); ++id)
    {
        for (int li = 0; li < L; ++li)
        {
            auto key = keyFor(data_.vectors[id].values, li);
            tables_[li][key].push_back(id);
        }
    }

    // Output debug information if LSH_DEBUG environment variable is set
    if (std::getenv("LSH_DEBUG"))
    {
        std::cerr << "[LSH-DIAG] w=" << w_
                  << " t_range=[" << t_min_ << "," << t_max_ << "]"
                  << " min_h_seen=" << (min_h_seen_ == LLONG_MAX ? 0 : min_h_seen_)
                  << " neg_h_count=" << neg_h_count_
                  << "\n";
    }
}

/**
 * Computes the hash key for a vector in a specific hash table
 * @param v Input vector to be hashed
 * @param li Index of the hash table
 * @return 64-bit hash key
 */
std::uint64_t LSH::keyFor(const std::vector<float> &v, int li) const
{
    // Initialize with FNV offset basis
    std::uint64_t h = 1469598103934665603ULL;
    const std::uint64_t prime = 1099511628211ULL; // FNV prime

    for (int j = 0; j < args.k; ++j)
    {
        // Compute dot product for projection
        double dot = 0.0;
        for (int d = 0; d < dim_; ++d)
            dot += a_[li][j][d] * v[d];

        // Apply LSH hash function: floor((a·v + t)/w)
        long long hj = (long long)std::floor((dot + t_[li][j]) / w_);

        // Combine hash values using FNV-1a hash
        h ^= (std::uint64_t)(hj * 11400714819323198485ull);
        h *= prime;

        // Update diagnostic counters
        if (hj < min_h_seen_)
            min_h_seen_ = hj;
        if (hj < 0)
            ++neg_h_count_;
    }
    return h;
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

    // Initialize performance tracking metrics
    double totalAF = 0.0;         // Approximation factor sum
    double totalRecall = 0.0;     // Recall sum
    double totalApproxTime = 0.0; // Total time for approximate search
    double totalTrueTime = 0.0;   // Total time for exact search
    int queryCount = (int)queries.vectors.size();

    // Process each query vector
    for (int qi = 0; qi < queryCount; ++qi)
    {
        const auto &q = queries.vectors[qi].values;

        // --- Start approximate search using LSH ---
        auto t0 = high_resolution_clock::now();

        // Collect candidate points from all L hash tables
        std::vector<int> candidates;
        for (int li = 0; li < args.L; ++li)
        {
            auto key = keyFor(q, li);        // Get hash key for query in table li
            auto it = tables_[li].find(key); // Look up bucket
            if (it != tables_[li].end())
                candidates.insert(candidates.end(), it->second.begin(), it->second.end());
        }

        // Calculate actual distances for candidate points
        std::vector<std::pair<double, int>> distApprox; // (distance, point_id) pairs
        distApprox.reserve(candidates.size());
        std::vector<int> rlist; // List for range search results

        // Process each candidate
        for (int id : candidates)
        {
            double d = l2(q, data_.vectors[id].values); // Compute true distance
            distApprox.emplace_back(d, id);
            // If doing range search and point is within radius R, add to results
            if (args.rangeSearch && d <= args.R)
                rlist.push_back(id);
        }

        // Find and sort top-N approximate nearest neighbors
        int N = std::min(args.N, (int)distApprox.size());
        if (N > 0)
        {
            // Partition to get top N elements
            std::nth_element(distApprox.begin(), distApprox.begin() + N, distApprox.end());
            // Sort the top N for output
            std::sort(distApprox.begin(), distApprox.begin() + N);
        }

        // Record time for approximate search
        auto t1 = high_resolution_clock::now();
        double tApprox = duration<double>(t1 - t0).count();
        totalApproxTime += tApprox;

        // --- Compute true nearest neighbors for comparison ---
        auto t2 = high_resolution_clock::now();
        std::vector<std::pair<double, int>> distTrue;
        distTrue.reserve(data_.vectors.size());
        // Calculate distances to all points
        for (const auto &v : data_.vectors)
            distTrue.emplace_back(l2(q, v.values), v.id);
        // Find and sort top N true nearest neighbors
        std::nth_element(distTrue.begin(), distTrue.begin() + N, distTrue.end());
        std::sort(distTrue.begin(), distTrue.begin() + N);
        auto t3 = high_resolution_clock::now();
        double tTrue = duration<double>(t3 - t2).count();
        totalTrueTime += tTrue;

        // --- Calculate quality metrics for this query ---
        double AFq = 0.0;     // Approximation factor for this query
        double recallq = 0.0; // Recall for this query
        int found = 0;        // Count of neighbors found

        // Compare approximate vs true neighbors
        for (int ni = 0; ni < N; ++ni)
        {
            // Calculate approximation factor
            double da = distApprox[ni].first; // Approximate distance
            double dt = distTrue[ni].first;   // True distance
            AFq += (dt > 0.0 ? da / dt : 1.0);

            // Check if approximate neighbor is among true top-N
            int approx_id = distApprox[ni].second;
            for (int j = 0; j < N; ++j)
                if (approx_id == distTrue[j].second)
                {
                    recallq += 1.0;
                    break;
                }
            found++;
        }

        // Normalize metrics by N if we found any neighbors
        if (N > 0)
        {
            AFq /= N;
            recallq /= N;
        }

        // Add to running totals
        totalAF += AFq;
        totalRecall += recallq;

        // --- Format and write results for this query ---
        out << "Query: " << qi << "\n";
        out << std::fixed << std::setprecision(6);
        for (int ni = 0; ni < N; ++ni)
        {
            out << "Nearest neighbor-" << (ni + 1) << ": " << distApprox[ni].second << "\n";
            out << "distanceApproximate: " << distApprox[ni].first << "\n";
            out << "distanceTrue: " << distTrue[ni].first << "\n";
        }
        out << "\nR-near neighbors:\n";
        for (int id : rlist)
            out << id << "\n";
        out << "\n";
    }

    // --- Calculate and output final summary statistics ---
    double avgAF = totalAF / queryCount;
    double avgRecall = totalRecall / queryCount;
    double avgApprox = totalApproxTime / queryCount;
    double avgTrue = totalTrueTime / queryCount;
    double qps = (avgApprox > 0) ? 1.0 / avgApprox : 0.0;

    // Write summary statistics
    out << "---- Summary (averages over queries) ----\n";
    out << "Average AF: " << avgAF << "\n";              // Approximation quality
    out << "Recall@N: " << avgRecall << "\n";            // Proportion of true neighbors found
    out << "QPS: " << qps << "\n";                       // Search throughput
    out << "tApproximateAverage: " << avgApprox << "\n"; // Average LSH search time
    out << "tTrueAverage: " << avgTrue << "\n";          // Average exact search time
}
