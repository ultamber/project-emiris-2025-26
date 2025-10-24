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
    out << "Hypercube\n";

    int M = args.M, probes = args.probes;
    double R = args.R;
    bool doRange = args.rangeSearch;

    double totalAF = 0.0, totalRecall = 0.0;
    double totalApproxTime = 0.0, totalTrueTime = 0.0;
    int queryCount = 0;

    auto startAll = high_resolution_clock::now();

    for (int qi = 0; qi < (int)queries.vectors.size(); ++qi) {
        const auto& q = queries.vectors[qi].values;

        auto startApprox = high_resolution_clock::now();

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

        int idA = -1;
        double bestA = std::numeric_limits<double>::infinity();
        std::vector<int> rlist;

        if (!cand.empty()) {
            for (int id : cand) {
                double d = l2(q, data_.vectors[id].values);
                if (doRange && d <= R) rlist.push_back(id);
                if (d < bestA) {
                    bestA = d;
                    idA = id;
                }
            }
        }

        auto endApprox = high_resolution_clock::now();
        double tApprox = duration<double>(endApprox - startApprox).count();

        // true nearest
        auto startTrue = high_resolution_clock::now();
        int idT = -1;
        double bestT = std::numeric_limits<double>::infinity();
        for (int id = 0; id < (int)data_.vectors.size(); ++id) {
            double d = l2(q, data_.vectors[id].values);
            if (d < bestT) {
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

        if (idA != -1) {
            totalAF += AF;
            totalRecall += recall;
            queryCount++;
        }

        out << "Query: " << qi << "\n";
        if (idA >= 0) {
            out << "Nearest neighbor-1: " << idA << "\n"
                << "distanceApproximate: " << bestA << "\n"
                << "distanceTrue: " << bestT << "\n";
        } else {
            out << "Nearest neighbor-1: -1\n"
                << "distanceApproximate: inf\n"
                << "distanceTrue: " << bestT << "\n";
        }
        out << "R-near neighbors:\n";
        for (int id : rlist) out << id << "\n";
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
