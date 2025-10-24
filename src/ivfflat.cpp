#include "ivfflat.hpp"
#include <random>
#include <limits>
#include <algorithm>
#include <chrono>
#include <iomanip>

double IVFFlat::l2(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

void IVFFlat::kmeans(const std::vector<std::vector<float>>& P, int k, int iters, unsigned seed) {
    int N = P.size(), D = P[0].size();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> uni(0, N - 1);

    centroids_.assign(k, std::vector<float>(D, 0));
    std::vector<int> init;
    init.reserve(k);
    for (int i = 0; i < k; ++i) {
        int id = uni(rng);
        while (std::find(init.begin(), init.end(), id) != init.end())
            id = uni(rng);
        init.push_back(id);
        centroids_[i] = P[id];
    }

    std::vector<int> assign(N, -1), cnt(k, 0);
    for (int it = 0; it < iters; ++it) {
        bool ch = false;
        for (int i = 0; i < N; ++i) {
            int best = -1;
            double bd = std::numeric_limits<double>::infinity();
            for (int c = 0; c < k; ++c) {
                double d = l2(P[i], centroids_[c]);
                if (d < bd) { bd = d; best = c; }
            }
            if (assign[i] != best) { assign[i] = best; ch = true; }
        }
        if (!ch) break;
        for (int c = 0; c < k; ++c) {
            std::fill(centroids_[c].begin(), centroids_[c].end(), 0);
            cnt[c] = 0;
        }
        for (int i = 0; i < N; ++i) {
            int c = assign[i];
            ++cnt[c];
            for (int d = 0; d < D; ++d)
                centroids_[c][d] += P[i][d];
        }
        for (int c = 0; c < k; ++c) {
            if (cnt[c])
                for (int d = 0; d < D; ++d)
                    centroids_[c][d] /= (float)cnt[c];
            else
                centroids_[c] = P[uni(rng)];
        }
    }

    lists_.assign(k, {});
    for (int i = 0; i < N; ++i)
        lists_[assign[i]].push_back(i);
}

void IVFFlat::buildIndex(const Dataset& data) {
    data_ = data;
    dim_ = data.dimension;
    k_ = args.kclusters;
    nprobe_ = args.nprobe;

    std::vector<std::vector<float>> P;
    P.reserve(data_.vectors.size());
    for (auto& v : data_.vectors)
        P.push_back(v.values);

    kmeans(P, k_, 25, (unsigned)args.seed);
}

void IVFFlat::search(const Dataset& queries, std::ofstream& out) {
    using namespace std::chrono;
    out << "IVFFlat\n\n";

    double R = args.R;
    bool doRange = args.rangeSearch;
    int N = args.N;

    double totalAF = 0.0, totalRecall = 0.0;
    double totalApproxTime = 0.0, totalTrueTime = 0.0;
    int Q = (int)queries.vectors.size();

    for (int qi = 0; qi < Q; ++qi) {
        const auto& q = queries.vectors[qi].values;

        // --- Approximate phase ---
        auto t0 = high_resolution_clock::now();

        // find nearest centroids
        std::vector<std::pair<double,int>> cds;
        cds.reserve(k_);
        for (int c = 0; c < k_; ++c)
            cds.emplace_back(l2(q, centroids_[c]), c);
        std::sort(cds.begin(), cds.end());
        int use = std::min(nprobe_, (int)cds.size());

        // gather candidates from closest clusters
        std::vector<int> candidates;
        for (int i = 0; i < use; ++i) {
            int cid = cds[i].second;
            candidates.insert(candidates.end(), lists_[cid].begin(), lists_[cid].end());
        }

        std::vector<std::pair<double,int>> distApprox;
        distApprox.reserve(candidates.size());
        std::vector<int> rlist;

        for (int id : candidates) {
            double d = l2(q, data_.vectors[id].values);
            distApprox.emplace_back(d, id);
            if (doRange && d <= R) rlist.push_back(id);
        }

        if (!distApprox.empty()) {
            int topN = std::min(N, (int)distApprox.size());
            std::nth_element(distApprox.begin(), distApprox.begin() + topN, distApprox.end());
            std::sort(distApprox.begin(), distApprox.begin() + topN);
            distApprox.resize(topN);
        }

        auto t1 = high_resolution_clock::now();
        double tApprox = duration<double>(t1 - t0).count();
        totalApproxTime += tApprox;

        // --- True exhaustive phase ---
        auto t2 = high_resolution_clock::now();
        std::vector<std::pair<double,int>> distTrue;
        distTrue.reserve(data_.vectors.size());
        for (const auto& v : data_.vectors)
            distTrue.emplace_back(l2(q, v.values), v.id);
        int topN = std::min(N, (int)distTrue.size());
        std::nth_element(distTrue.begin(), distTrue.begin() + topN, distTrue.end());
        std::sort(distTrue.begin(), distTrue.begin() + topN);
        distTrue.resize(topN);
        auto t3 = high_resolution_clock::now();
        double tTrue = duration<double>(t3 - t2).count();
        totalTrueTime += tTrue;

        // --- Metrics ---
        double AFq = 0.0;
        double recallq = 0.0;
        for (int ni = 0; ni < (int)distApprox.size(); ++ni) {
            double da = distApprox[ni].first;
            double dt = distTrue[ni].first;
            AFq += (dt > 0.0) ? da / dt : 1.0;
            // check if approx id in true topN
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
        out << "Average AF: " << AFq << "\n";
        out << "Recall@N: " << recallq << "\n";
        out << "QPS: " << (tApprox > 0.0 ? 1.0 / tApprox : 0.0) << "\n";
        out << "tApproximateAverage: " << tApprox << "\n";
        out << "tTrueAverage: " << tTrue << "\n\n";
    }

    // --- Summary over all queries ---
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
