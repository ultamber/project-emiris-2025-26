#include "hypercube.hpp"
#include <random>
#include <limits>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <chrono>
#include <iomanip>

double Hypercube::l2(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

void Hypercube::buildIndex(const Dataset& data) {
    data_ = data;
    dim_ = data.dimension;
    int kproj = args.kproj;

    std::mt19937_64 rng(args.seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    proj_.assign(kproj, std::vector<float>(dim_));
    for (int j = 0; j < kproj; ++j)
        for (int d = 0; d < dim_; ++d)
            proj_[j][d] = (float)normal(rng);

    cube_.clear();
    for (int id = 0; id < (int)data_.vectors.size(); ++id)
        cube_[vertexOf(data_.vectors[id].values)].push_back(id);
}

std::uint64_t Hypercube::vertexOf(const std::vector<float>& v) const {
    std::uint64_t key = 0;
    for (int j = 0; j < args.kproj; ++j) {
        double dot = 0.0;
        for (int d = 0; d < dim_; ++d)
            dot += proj_[j][d] * v[d];
        if (dot >= 0.0)
            key |= (1ULL << j);
    }
    return key;
}

// simple increasing-Hamming enumeration up to maxProbes
std::vector<std::uint64_t> Hypercube::probesList(std::uint64_t base, int kproj, int maxProbes) const {
    std::vector<std::uint64_t> out{base};
    if ((int)out.size() >= maxProbes) return out;
    for (int h = 1; h <= kproj && (int)out.size() < maxProbes; ++h) {
        std::vector<int> idx(h);
        std::iota(idx.begin(), idx.end(), 0);
        while (true) {
            std::uint64_t mask = 0;
            for (int i : idx) mask |= (1ULL << i);
            out.push_back(base ^ mask);
            if ((int)out.size() >= maxProbes) break;
            int i;
            for (i = h - 1; i >= 0 && idx[i] == i + kproj - h; --i);
            if (i < 0) break;
            ++idx[i];
            for (int j = i + 1; j < h; ++j) idx[j] = idx[j - 1] + 1;
        }
    }
    return out;
}

void Hypercube::search(const Dataset& queries, std::ofstream& out) {
    using namespace std::chrono;
    out << "Hypercube\n\n";

    int M = args.M;
    int probes = args.probes;
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

        std::unordered_set<int> cand;
        cand.reserve(1024);

        auto base = vertexOf(q);
        auto plist = probesList(base, args.kproj, probes);

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

        std::vector<std::pair<double,int>> distApprox;
        distApprox.reserve(cand.size());
        std::vector<int> rlist;

        for (int id : cand) {
            double d = l2(q, data_.vectors[id].values);
            distApprox.emplace_back(d, id);
            if (doRange && d <= R) rlist.push_back(id);
        }

        if (!distApprox.empty()) {
            int topN = std::min(N, (int)distApprox.size());
            std::nth_element(distApprox.begin(), distApprox.begin()+topN, distApprox.end());
            std::sort(distApprox.begin(), distApprox.begin()+topN);
            distApprox.resize(topN);
        }

        auto t1 = high_resolution_clock::now();
        double tApprox = duration<double>(t1 - t0).count();
        totalApproxTime += tApprox;

        // --- True phase ---
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
        out << "Average AF: " << AFq << "\n";
        out << "Recall@N: " << recallq << "\n";
        out << "QPS: " << (tApprox > 0.0 ? 1.0 / tApprox : 0.0) << "\n";
        out << "tApproximateAverage: " << tApprox << "\n";
        out << "tTrueAverage: " << tTrue << "\n\n";
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
