#include "hypercube.hpp"
#include <random>
#include <limits>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <chrono>
#include <iomanip>

/**
 * Computes the L2 (Euclidean) distance between two vectors
 * @param a First vector
 * @param b Second vector
 * @return The Euclidean distance between vectors a and b
 */
double Hypercube::l2(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

/**
 * Builds the Hypercube index for the input dataset
 * @param data Input dataset to be indexed
 */
void Hypercube::buildIndex(const Dataset& data) {
    // Store dataset and its dimension
    data_ = data;
    dim_ = data.dimension;
    int kproj = args.kproj;  // Number of projections (bits in vertex label)

    // Initialize random number generator for projections
    std::mt19937_64 rng(args.seed);
    std::normal_distribution<double> normal(0.0, 1.0);  // For random projections

    // Generate random projection vectors
    proj_.assign(kproj, std::vector<float>(dim_));
    for (int j = 0; j < kproj; ++j)
        for (int d = 0; d < dim_; ++d)
            proj_[j][d] = (float)normal(rng);

    // Build the hypercube by mapping each point to a vertex
    cube_.clear();
    for (int id = 0; id < (int)data_.vectors.size(); ++id)
        cube_[vertexOf(data_.vectors[id].values)].push_back(id);
}

/**
 * Maps a vector to a vertex in the hypercube
 * @param v Input vector
 * @return 64-bit vertex label where each bit represents a projection result
 */
std::uint64_t Hypercube::vertexOf(const std::vector<float>& v) const {
    std::uint64_t key = 0;
    for (int j = 0; j < args.kproj; ++j) {
        // Compute dot product with projection vector
        double dot = 0.0;
        for (int d = 0; d < dim_; ++d)
            dot += proj_[j][d] * v[d];
        // Set bit j based on projection sign
        if (dot >= 0.0)
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
std::vector<std::uint64_t> Hypercube::probesList(std::uint64_t base, int kproj, int maxProbes) const {
    std::vector<std::uint64_t> out{base};
    if ((int)out.size() >= maxProbes) return out;

    // Generate vertices at increasing Hamming distances
    for (int h = 1; h <= kproj && (int)out.size() < maxProbes; ++h) {
        // Initialize combination indices
        std::vector<int> idx(h);
        std::iota(idx.begin(), idx.end(), 0);

        while (true) {
            // Generate vertex by flipping bits at current positions
            std::uint64_t mask = 0;
            for (int i : idx) mask |= (1ULL << i);
            out.push_back(base ^ mask);
            
            if ((int)out.size() >= maxProbes) break;

            // Generate next combination in lexicographic order
            int i;
            for (i = h - 1; i >= 0 && idx[i] == i + kproj - h; --i);
            if (i < 0) break;
            ++idx[i];
            for (int j = i + 1; j < h; ++j) idx[j] = idx[j - 1] + 1;
        }
    }
    return out;
}

/**
 * Performs Hypercube search for all queries
 * @param queries Query dataset
 * @param out Output file stream for results
 */
void Hypercube::search(const Dataset& queries, std::ofstream& out) {
    using namespace std::chrono;
    out << "Hypercube\n\n";

    // Extract search parameters
    int M = args.M;           // Maximum candidates to consider
    int probes = args.probes; // Maximum vertices to probe
    double R = args.R;        // Range search radius
    bool doRange = args.rangeSearch;
    int N = args.N;           // Number of nearest neighbors to find

    // Initialize performance tracking metrics
    double totalAF = 0.0, totalRecall = 0.0;
    double totalApproxTime = 0.0, totalTrueTime = 0.0;
    int Q = (int)queries.vectors.size();

    // Process each query
    for (int qi = 0; qi < Q; ++qi) {
        const auto& q = queries.vectors[qi].values;

        // --- Approximate search phase ---
        auto t0 = high_resolution_clock::now();

        // Collect candidate points from nearby vertices
        std::unordered_set<int> cand;
        cand.reserve(1024);

        // Generate list of vertices to probe
        auto base = vertexOf(q);
        auto plist = probesList(base, args.kproj, probes);

        // Collect points from each vertex until M candidates found
        for (auto vtx : plist) {
            auto it = cube_.find(vtx);
            if (it != cube_.end()) {
                for (int id : it->second) {
                    cand.insert(id);
                    if ((int)cand.size() >= M) break;
                }
            }
            if ((int)cand.size() >= M) break;
        }

        // Calculate distances to candidates
        std::vector<std::pair<double,int>> distApprox;
        distApprox.reserve(cand.size());
        std::vector<int> rlist;  // For range search results

        for (int id : cand) {
            double d = l2(q, data_.vectors[id].values);
            distApprox.emplace_back(d, id);
            if (doRange && d <= R) rlist.push_back(id);
        }

        // Sort top-N approximate neighbors
        if (!distApprox.empty()) {
            int topN = std::min(N, (int)distApprox.size());
            std::nth_element(distApprox.begin(), distApprox.begin()+topN, distApprox.end());
            std::sort(distApprox.begin(), distApprox.begin()+topN);
            distApprox.resize(topN);
        }

        // Record approximate search time
        auto t1 = high_resolution_clock::now();
        double tApprox = duration<double>(t1 - t0).count();
        totalApproxTime += tApprox;

        // --- True search phase ---
        auto t2 = high_resolution_clock::now();
        std::vector<std::pair<double,int>> distTrue;
        distTrue.reserve(data_.vectors.size());
        for (const auto &v : data_.vectors)
            distTrue.emplace_back(l2(q, v.values), v.id);
        int topN = std::min(N, (int)distTrue.size());
        std::nth_element(distTrue.begin(), distTrue.begin()+topN, distTrue.end());
        std::sort(distTrue.begin(), distTrue.begin()+topN);
        distTrue.resize(topN);
        auto t3 = high_resolution_clock::now();
        double tTrue = duration<double>(t3 - t2).count();
        totalTrueTime += tTrue;

        // --- Compute metrics ---
        double AFq = 0.0, recallq = 0.0;
        for (int ni = 0; ni < (int)distApprox.size(); ++ni) {
            double da = distApprox[ni].first;
            double dt = distTrue[ni].first;
            AFq += (dt > 0.0) ? da / dt : 1.0;

            // recall@N: check if this approx id is in true topN
            for (auto &p : distTrue) {
                if (p.second == distApprox[ni].second) {
                    recallq += 1.0;
                    break;
                }
            }
        }

        if (N > 0 && !distApprox.empty()) {
            AFq /= distApprox.size();
            recallq /= N;
        }

        totalAF += AFq;
        totalRecall += recallq;

        // --- Output per query ---
        out << "Query: " << qi << "\n";
        out << std::fixed << std::setprecision(6);
        for (int ni = 0; ni < (int)distApprox.size(); ++ni) {
            out << "Nearest neighbor-" << (ni+1) << ": " << distApprox[ni].second << "\n";
            out << "distanceApproximate: " << distApprox[ni].first << "\n";
            out << "distanceTrue: " << distTrue[ni].first << "\n";
        }
        out << "\nR-near neighbors:\n";
        for (int id : rlist) out << id << "\n";
        out << "\n";
    }

    // --- Summary ---
    double avgAF = totalAF / Q;
    double avgRecall = totalRecall / Q;
    double avgApprox = totalApproxTime / Q;
    double avgTrue = totalTrueTime / Q;
    double qpsOverall = (avgApprox > 0) ? 1.0 / avgApprox : 0.0;

    out << "---- Summary (averages over queries) ----\n";
    out << std::fixed << std::setprecision(6)
        << "Average AF: " << avgAF << "\n"
        << "Recall@N: " << avgRecall << "\n"
        << "QPS: " << qpsOverall << "\n"
        << "tApproximateAverage: " << avgApprox << "\n"
        << "tTrueAverage: " << avgTrue << "\n";
}
