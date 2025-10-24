#include "ivfpq.hpp"
#include <random>
#include <limits>
#include <algorithm>
#include <chrono>
#include <iomanip>

double IVFPQ::l2(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

void IVFPQ::kmeans(const std::vector<std::vector<float>>& P, int k, int iters, unsigned seed,
                   std::vector<std::vector<float>>& C, std::vector<int>* outAssign) {
    int N = P.size(), D = P[0].size();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> uni(0, N - 1);
    C.assign(k, std::vector<float>(D, 0));
    std::vector<int> init;
    init.reserve(k);
    for (int i = 0; i < k; ++i) {
        int id = uni(rng);
        while (std::find(init.begin(), init.end(), id) != init.end()) id = uni(rng);
        init.push_back(id);
        C[i] = P[id];
    }

    std::vector<int> assign(N, -1), cnt(k, 0);
    for (int it = 0; it < iters; ++it) {
        bool ch = false;
        for (int i = 0; i < N; ++i) {
            int best = -1;
            double bd = std::numeric_limits<double>::infinity();
            for (int c = 0; c < k; ++c) {
                double d = l2(P[i], C[c]);
                if (d < bd) { bd = d; best = c; }
            }
            if (assign[i] != best) { assign[i] = best; ch = true; }
        }
        if (!ch) break;
        for (int c = 0; c < k; ++c) {
            std::fill(C[c].begin(), C[c].end(), 0);
            cnt[c] = 0;
        }
        for (int i = 0; i < N; ++i) {
            int c = assign[i];
            ++cnt[c];
            for (int d = 0; d < D; ++d)
                C[c][d] += P[i][d];
        }
        for (int c = 0; c < k; ++c) {
            if (cnt[c])
                for (int d = 0; d < D; ++d)
                    C[c][d] /= (float)cnt[c];
            else
                C[c] = P[uni(rng)];
        }
    }
    if (outAssign) *outAssign = std::move(assign);
}

void IVFPQ::makeResiduals(const Dataset& data, const std::vector<int>& assign,
                          std::vector<std::vector<float>>& residuals) {
    int N = data.vectors.size(), D = data.dimension;
    residuals.assign(N, std::vector<float>(D, 0));
    for (int i = 0; i < N; ++i) {
        int c = assign[i];
        for (int d = 0; d < D; ++d)
            residuals[i][d] = data.vectors[i].values[d] - centroids_[c][d];
    }
}

void IVFPQ::trainPQ(const std::vector<std::vector<float>>& R) {
    int D = dim_, M = M_;
    subdim_.assign(M, D / M);
    for (int i = 0; i < D % M; ++i) ++subdim_[i];
    codebooks_.assign(M, {});
    unsigned seed = (unsigned)args.seed;

    int offset = 0;
    for (int m = 0; m < M; ++m) {
        int sd = subdim_[m];
        std::vector<std::vector<float>> sub;
        sub.reserve(R.size());
        for (auto& r : R)
            sub.emplace_back(r.begin() + offset, r.begin() + offset + sd);

        std::vector<std::vector<float>> cb;
        kmeans(sub, Ks_, 20, seed + m, cb, nullptr);
        codebooks_[m] = std::move(cb);
        offset += sd;
    }
}

void IVFPQ::encodeAll(const std::vector<std::vector<float>>& R) {
    int N = R.size(), M = M_;
    codes_.assign(N, std::vector<uint8_t>(M, 0));
    int offset0 = 0;
    for (int m = 0; m < M; ++m) {
        int sd = subdim_[m];
        for (int i = 0; i < N; ++i) {
            int best = 0;
            double bd = std::numeric_limits<double>::infinity();
            for (int k = 0; k < Ks_; ++k) {
                double s = 0;
                for (int d = 0; d < sd; ++d) {
                    double diff = R[i][offset0 + d] - codebooks_[m][k][d];
                    s += diff * diff;
                }
                if (s < bd) { bd = s; best = k; }
            }
            codes_[i][m] = (uint8_t)best;
        }
        offset0 += sd;
    }
}

void IVFPQ::buildIndex(const Dataset& data) {
    data_ = data;
    dim_ = data.dimension;
    k_ = args.kclusters;
    nprobe_ = args.nprobe;
    M_ = args.Msubvectors;
    Ks_ = 1 << args.nbits;

    std::vector<std::vector<float>> P;
    P.reserve(data_.vectors.size());
    for (auto& v : data_.vectors)
        P.push_back(v.values);

    std::vector<int> assign;
    kmeans(P, k_, 50, (unsigned)args.seed, centroids_, &assign);
    lists_.assign(k_, {});
    for (int i = 0; i < (int)assign.size(); ++i)
        lists_[assign[i]].push_back(i);

    std::vector<std::vector<float>> R;
    makeResiduals(data_, assign, R);
    trainPQ(R);
    encodeAll(R);
}

void IVFPQ::search(const Dataset& queries, std::ofstream& out) {
    using namespace std::chrono;
    out << "IVFPQ\n";
    double Rrad = args.R;
    bool doRange = args.rangeSearch;

    double totalAF = 0.0, totalRecall = 0.0;
    double totalApproxTime = 0.0, totalTrueTime = 0.0;
    int queryCount = 0;

    auto startAll = high_resolution_clock::now();

    for (int qi = 0; qi < (int)queries.vectors.size(); ++qi) {
        const auto& q = queries.vectors[qi].values;
        auto startApprox = high_resolution_clock::now();

        // nearest nprobe centroids
        std::vector<std::pair<double, int>> cds;
        cds.reserve(k_);
        for (int c = 0; c < k_; ++c)
            cds.emplace_back(l2(q, centroids_[c]), c);
        std::sort(cds.begin(), cds.end());
        int use = std::min(nprobe_, (int)cds.size());

        int idA = -1;
        double bestA = std::numeric_limits<double>::infinity();
        std::vector<int> rlist;

        for (int pi = 0; pi < use; ++pi) {
            int cid = cds[pi].second;

            // query residual wrt this centroid
            std::vector<float> rq(dim_);
            for (int d = 0; d < dim_; ++d)
                rq[d] = q[d] - centroids_[cid][d];

            // LUTs per subspace
            std::vector<std::vector<double>> LUT(M_, std::vector<double>(Ks_, 0.0));
            int off = 0;
            for (int m = 0; m < M_; ++m) {
                int sd = subdim_[m];
                for (int k = 0; k < Ks_; ++k) {
                    double s = 0;
                    for (int d = 0; d < sd; ++d) {
                        double diff = rq[off + d] - codebooks_[m][k][d];
                        s += diff * diff;
                    }
                    LUT[m][k] = s;
                }
                off += sd;
            }

            // scan the list using ADC
            for (int id : lists_[cid]) {
                double da = 0;
                for (int m = 0; m < M_; ++m)
                    da += LUT[m][codes_[id][m]];
                if (doRange && da <= Rrad) rlist.push_back(id);
                if (da < bestA) { bestA = da; idA = id; }
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
            if (d < bestT) { bestT = d; idT = id; }
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
