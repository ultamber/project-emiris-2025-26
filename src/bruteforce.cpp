#include "bruteforce.hpp"
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <cmath>

BruteForce::BruteForce(const Arguments& args) : cfg(args) {}

void BruteForce::buildIndex(const Dataset& data) {
    // No index to build for brute-force; keep pointer/reference for search
    indexedData = &data;
}

double BruteForce::euclidean_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    size_t D = a.size();
    for (size_t i = 0; i < D; ++i) {
        double diff = double(a[i]) - double(b[i]);
        s += diff * diff;
    }
    return std::sqrt(s);
}

void BruteForce::search(const Dataset& queries, std::ofstream& out) {
    if (!indexedData) {
        throw std::runtime_error("BruteForce: index not built (call buildIndex first)");
    }

    const auto& base = indexedData->vectors;
    int Q = static_cast<int>(queries.vectors.size());
    if (Q == 0) return;

    // global summary accumulators
    double total_AF = 0.0;
    double total_recall = 0.0;
    double total_approx_time = 0.0;
    double total_true_time = 0.0;

    out << "BruteForce\n\n";

    for (int qi = 0; qi < Q; ++qi) {
        const auto& qvec = queries.vectors[qi].values;

        // measure brute-force time (we treat this as both 'approx' and 'true' for baseline)
        auto t0 = std::chrono::high_resolution_clock::now();

        // compute distances to all base vectors
        std::vector<std::pair<double,int>> dist_id;
        dist_id.reserve(base.size());
        for (const auto& item : base) {
            double d = euclidean_l2(qvec, item.values);
            dist_id.emplace_back(d, item.id);
        }

        // sort / partial-select top N
        int N = std::min(cfg.N, (int)dist_id.size());
        if (N <= 0) N = 1;
        std::nth_element(dist_id.begin(), dist_id.begin() + N, dist_id.end(),
                         [](const auto& a, const auto& b){ return a.first < b.first; });
        std::sort(dist_id.begin(), dist_id.begin() + N,
                  [](const auto& a, const auto& b){ return a.first < b.first; });

        // collect R-near neighbors if requested
        std::vector<int> rnear_ids;
        if (cfg.rangeSearch) {
            for (const auto &p : dist_id) {
                if (p.first <= cfg.R + 1e-12) rnear_ids.push_back(p.second);
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        // For brute-force baseline approximate == true
        double approx_time = elapsed;
        double true_time = elapsed;
        total_approx_time += approx_time;
        total_true_time += true_time;

        // Output format per query
        out << "Query: " << qi << "\n";
        for (int ni = 0; ni < N; ++ni) {
            out << "Nearest neighbor-" << (ni+1) << ": " << dist_id[ni].second << "\n";
            out << "distanceApproximate: " << std::fixed << std::setprecision(6) << dist_id[ni].first << "\n";
            out << "distanceTrue: " << std::fixed << std::setprecision(6) << dist_id[ni].first << "\n";
        }
        out << "\n";

        out << "R-near neighbors:\n";
        for (int id : rnear_ids) out << id << "\n";
        out << "\n";

        // Average AF (approx / true). For brute force this is 1.0 but compute defensively.
        double AF_q = 0.0;
        for (int ni = 0; ni < N; ++ni) {
            double approxd = dist_id[ni].first;
            double trued = dist_id[ni].first;
            AF_q += (trued > 0.0) ? (approxd / trued) : 1.0;
        }
        AF_q /= N;
        total_AF += AF_q;

        // Recall@N for brute force is 1.0 because we computed exact neighbors
        double recall_q = 1.0;
        total_recall += recall_q;

        out << "Average AF: " << std::fixed << std::setprecision(6) << AF_q << "\n";
        out << "Recall@N: " << std::fixed << std::setprecision(6) << recall_q << "\n";
        out << "QPS: " << std::fixed << std::setprecision(6)
            << ((approx_time > 0.0) ? (1.0 / approx_time) : 0.0) << "\n";
        out << "tApproximateAverage: " << std::fixed << std::setprecision(6) << approx_time << "\n";
        out << "tTrueAverage: " << std::fixed << std::setprecision(6) << true_time << "\n";
        out << "\n";
    }

    // overall summary
    double avg_AF = total_AF / Q;
    double avg_recall = total_recall / Q;
    double avg_approx_time = total_approx_time / Q;
    double avg_true_time = total_true_time / Q;
    double qps_overall = (avg_approx_time > 0.0) ? (1.0 / avg_approx_time) : 0.0;

    out << "---- Summary (averages over queries) ----\n";
    out << "Average AF: " << std::fixed << std::setprecision(6) << avg_AF << "\n";
    out << "Recall@N: " << std::fixed << std::setprecision(6) << avg_recall << "\n";
    out << "QPS: " << std::fixed << std::setprecision(6) << qps_overall << "\n";
    out << "tApproximateAverage: " << std::fixed << std::setprecision(6) << avg_approx_time << "\n";
    out << "tTrueAverage: " << std::fixed << std::setprecision(6) << avg_true_time << "\n";
    out << "\n";
}
