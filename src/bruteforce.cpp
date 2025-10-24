#include "bruteforce.hpp"
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <cmath>

BruteForce::BruteForce(const Arguments& args) : cfg(args) {}

void BruteForce::buildIndex(const Dataset& data) {
    indexedData = &data;
}

double BruteForce::euclidean_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = double(a[i]) - double(b[i]);
        s += diff * diff;
    }
    return std::sqrt(s);
}

void BruteForce::search(const Dataset& queries, std::ofstream& out) {
    using namespace std::chrono;
    if (!indexedData) throw std::runtime_error("BruteForce: index not built");

    const auto& base = indexedData->vectors;
    int Q = (int)queries.vectors.size();
    int N = std::max(1, cfg.N);

    double total_AF = 0.0;
    double total_recall = 0.0;
    double total_time = 0.0;

    out << "BruteForce\n\n";

    for (int qi = 0; qi < Q; ++qi) {
        const auto& q = queries.vectors[qi].values;

        auto t0 = high_resolution_clock::now();

        // Compute distances to all dataset vectors
        std::vector<std::pair<double,int>> dist;
        dist.reserve(base.size());
        for (const auto& v : base)
            dist.emplace_back(euclidean_l2(q, v.values), v.id);

        // Sort and keep top N
        std::nth_element(dist.begin(), dist.begin() + N, dist.end());
        std::sort(dist.begin(), dist.begin() + N);

        // Gather R-near neighbors if requested
        std::vector<int> rnear;
        if (cfg.rangeSearch) {
            for (auto &p : dist)
                if (p.first <= cfg.R + 1e-12) rnear.push_back(p.second);
        }

        auto t1 = high_resolution_clock::now();
        double tQuery = duration<double>(t1 - t0).count();
        total_time += tQuery;

        // --- Metrics ---
        double AFq = 1.0;       // all exact => AF = 1
        double recallq = 1.0;   // all true => Recall@N = 1

        total_AF += AFq;
        total_recall += recallq;

        // --- Output per query ---
        out << "Query: " << qi << "\n";
        out << std::fixed << std::setprecision(6);
        for (int ni = 0; ni < N; ++ni) {
            out << "Nearest neighbor-" << (ni + 1) << ": " << dist[ni].second << "\n";
            out << "distanceApproximate: " << dist[ni].first << "\n";
            out << "distanceTrue: " << dist[ni].first << "\n";
        }

        out << "\nR-near neighbors:\n";
        for (int id : rnear) out << id << "\n";
        out << "\n";

        out << "Average AF: " << AFq << "\n";
        out << "Recall@N: " << recallq << "\n";
        out << "QPS: " << (tQuery > 0.0 ? 1.0 / tQuery : 0.0) << "\n";
        out << "tApproximateAverage: " << tQuery << "\n";
        out << "tTrueAverage: " << tQuery << "\n\n";
    }

    // --- Summary over all queries ---
    double avg_AF = total_AF / Q;
    double avg_recall = total_recall / Q;
    double avg_time = total_time / Q;
    double qps = (avg_time > 0.0) ? 1.0 / avg_time : 0.0;

    out << "---- Summary (averages over queries) ----\n";
    out << std::fixed << std::setprecision(6)
        << "Average AF: " << avg_AF << "\n"
        << "Recall@N: " << avg_recall << "\n"
        << "QPS: " << qps << "\n"
        << "tApproximateAverage: " << avg_time << "\n"
        << "tTrueAverage: " << avg_time << "\n";
}
