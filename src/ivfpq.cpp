#include "ivfpq.hpp"
#include <random>
#include <limits>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <unordered_set>

double IVFPQ::l2(const std::vector<float> &a, const std::vector<float> &b)
{
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

void IVFPQ::kmeans(const std::vector<std::vector<float>> &P, int k, int iters, unsigned seed,
                   std::vector<std::vector<float>> &C, std::vector<int> *outAssign)
{
    int N = P.size(), D = P[0].size();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> uni(0, N - 1);
    C.assign(k, std::vector<float>(D, 0));
    std::vector<int> init;
    init.reserve(k);
    for (int i = 0; i < k; ++i)
    {
        int id = uni(rng);
        while (std::find(init.begin(), init.end(), id) != init.end())
            id = uni(rng);
        init.push_back(id);
        C[i] = P[id];
    }

    std::vector<int> assign(N, -1), cnt(k, 0);
    for (int it = 0; it < iters; ++it)
    {
        bool ch = false;
        for (int i = 0; i < N; ++i)
        {
            int best = -1;
            double bd = std::numeric_limits<double>::infinity();
            for (int c = 0; c < k; ++c)
            {
                double d = l2(P[i], C[c]);
                if (d < bd)
                {
                    bd = d;
                    best = c;
                }
            }
            if (assign[i] != best)
            {
                assign[i] = best;
                ch = true;
            }
        }
        if (!ch)
            break;
        for (int c = 0; c < k; ++c)
        {
            std::fill(C[c].begin(), C[c].end(), 0);
            cnt[c] = 0;
        }
        for (int i = 0; i < N; ++i)
        {
            int c = assign[i];
            ++cnt[c];
            for (int d = 0; d < D; ++d)
                C[c][d] += P[i][d];
        }
        for (int c = 0; c < k; ++c)
        {
            if (cnt[c])
                for (int d = 0; d < D; ++d)
                    C[c][d] /= (float)cnt[c];
            else
                C[c] = P[uni(rng)];
        }
    }
    if (outAssign)
        *outAssign = std::move(assign);
}

void IVFPQ::makeResiduals(const Dataset &data, const std::vector<int> &assign,
                          std::vector<std::vector<float>> &residuals)
{
    int N = data.vectors.size(), D = data.dimension;
    residuals.assign(N, std::vector<float>(D, 0));
    for (int i = 0; i < N; ++i)
    {
        int c = assign[i];
        for (int d = 0; d < D; ++d)
            residuals[i][d] = data.vectors[i].values[d] - centroids_[c][d];
    }
}

void IVFPQ::trainPQ(const std::vector<std::vector<float>> &R)
{
    int D = dim_, M = M_;
    subdim_.assign(M, D / M);
    for (int i = 0; i < D % M; ++i)
        ++subdim_[i];
    codebooks_.assign(M, {});
    unsigned seed = (unsigned)args.seed;

    int offset = 0;
    for (int m = 0; m < M; ++m)
    {
        int sd = subdim_[m];
        std::vector<std::vector<float>> sub;
        sub.reserve(R.size());
        for (auto &r : R)
            sub.emplace_back(r.begin() + offset, r.begin() + offset + sd);

        std::vector<std::vector<float>> cb;
        kmeans(sub, Ks_, 20, seed + m, cb, nullptr);
        codebooks_[m] = std::move(cb);
        offset += sd;
    }
}

void IVFPQ::encodeAll(const std::vector<std::vector<float>> &R)
{
    int N = R.size(), M = M_;
    codes_.assign(N, std::vector<uint8_t>(M, 0));
    int offset0 = 0;
    for (int m = 0; m < M; ++m)
    {
        int sd = subdim_[m];
        for (int i = 0; i < N; ++i)
        {
            int best = 0;
            double bd = std::numeric_limits<double>::infinity();
            for (int k = 0; k < Ks_; ++k)
            {
                double s = 0;
                for (int d = 0; d < sd; ++d)
                {
                    double diff = R[i][offset0 + d] - codebooks_[m][k][d];
                    s += diff * diff;
                }
                if (s < bd)
                {
                    bd = s;
                    best = k;
                }
            }
            codes_[i][m] = (uint8_t)best;
        }
        offset0 += sd;
    }
}

void IVFPQ::buildIndex(const Dataset &data)
{
    data_ = data;
    dim_  = data.dimension;

    k_       = args.kclusters;     // #coarse centroids (IVF)
    nprobe_  = args.nprobe;        // how many inverted lists to scan at query
    M_       = args.Msubvectors;   // #subquantizers
    Ks_      = 1 << args.nbits;    // codebook size per subquantizer

    const int N = (int)data_.vectors.size();
    if (N == 0) {
        centroids_.clear();
        lists_.clear();
        return;
    }

    // --- 1) build coarse quantizer on a subset X' ~ sqrt(n) (slide) ---
    int trainN = std::max(k_, (int)std::sqrt((double)N));
    std::mt19937_64 rng(args.seed);
    std::vector<int> idx(N); std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(trainN);

    std::vector<std::vector<float>> Ptrain;
    Ptrain.reserve(trainN);
    for (int id : idx) Ptrain.push_back(data_.vectors[id].values);

    std::vector<int> coarseAssign;  // assignment of Ptrain (we don't really need it)
    kmeans(Ptrain, k_, 40, (unsigned)args.seed, centroids_, nullptr);

    // --- 2) assign ALL points to inverted lists (IVF build) ---
    lists_.assign(k_, {});
    std::vector<int> fullAssign(N, -1);
    for (int i = 0; i < N; ++i) {
        const auto &x = data_.vectors[i].values;
        int best = 0;
        double bd = l2(x, centroids_[0]);
        for (int c = 1; c < k_; ++c) {
            double d = l2(x, centroids_[c]);
            if (d < bd) { bd = d; best = c; }
        }
        lists_[best].push_back(i);
        fullAssign[i] = best;
    }

    // --- 3) residuals r(x) = x - c(x) for ALL x ---
    std::vector<std::vector<float>> R;
    makeResiduals(data_, fullAssign, R);

    // --- 4) train product quantizer on residuals (slide steps 4–7) ---
    trainPQ(R);

    // --- 5) encode ALL residuals with trained PQ ---
    encodeAll(R);
}


void IVFPQ::search(const Dataset& queries, std::ofstream& out) {
    using namespace std::chrono;
    out << "IVFPQ\n\n";

    const double Rrad   = args.R;
    const bool   doRange = args.rangeSearch;
    const int    Nret    = std::max(1, args.N);
    const int    rerankT = 100;   // top T to re-rank exactly

    double totalAF = 0.0, totalRecall = 0.0;
    double totalApproxTime = 0.0, totalTrueTime = 0.0;
    const int Q = (int)queries.vectors.size();

    for (int qi = 0; qi < Q; ++qi) {
        const auto& q = queries.vectors[qi].values;

        // --- Approximate (IVF + PQ-ADC) ---
        auto t0 = high_resolution_clock::now();

        // 1) score all coarse centroids
        std::vector<std::pair<double,int>> cds;
        cds.reserve(k_);
        for (int c = 0; c < k_; ++c) {
            double s = 0.0;
            for (int d = 0; d < dim_; ++d) {
                double diff = q[d] - centroids_[c][d];
                s += diff * diff;
            }
            cds.emplace_back(s, c);   // squared distance is fine
        }
        std::sort(cds.begin(), cds.end());
        const int use = std::min(nprobe_, (int)cds.size());

        std::vector<std::pair<double,int>> approx; // (adc_dist_sqr, id)
        std::vector<int> rlist;

        for (int pi = 0; pi < use; ++pi) {
            int cid = cds[pi].second;

            // query residual: q - c(cid)
            std::vector<float> rq(dim_);
            for (int d = 0; d < dim_; ++d)
                rq[d] = q[d] - centroids_[cid][d];

            // build LUTs for this residual
            std::vector<std::vector<double>> LUT(M_, std::vector<double>(Ks_, 0.0));
            int off = 0;
            for (int m = 0; m < M_; ++m) {
                int sd = subdim_[m];
                for (int k = 0; k < Ks_; ++k) {
                    double s = 0.0;
                    for (int d = 0; d < sd; ++d) {
                        double diff = rq[off + d] - codebooks_[m][k][d];
                        s += diff * diff;
                    }
                    LUT[m][k] = s;     // store squared distance
                }
                off += sd;
            }

            // scan candidates in this inverted list
            for (int id : lists_[cid]) {
                double adc_sqr = 0.0;
                for (int m = 0; m < M_; ++m)
                    adc_sqr += LUT[m][ codes_[id][m] ];
                approx.emplace_back(adc_sqr, id);

                // range in PQ space (optional)
                if (doRange) {
                    double adc = std::sqrt(adc_sqr);
                    if (adc <= Rrad) rlist.push_back(id);
                }
            }
        }

        // 2) keep top-T (we'll re-rank T exactly)
        if (!approx.empty()) {
            int T = std::max(Nret, rerankT);
            T = std::min(T, (int)approx.size());
            std::nth_element(approx.begin(), approx.begin()+T, approx.end());
            std::sort(approx.begin(), approx.begin()+T);
            approx.resize(T);
        }

        // 3) exact re-ranking on original vectors
        for (auto &p : approx)
            p.first = l2(q, data_.vectors[p.second].values);
        if (!approx.empty()) {
            int keep = std::min(Nret, (int)approx.size());
            std::nth_element(approx.begin(), approx.begin()+keep, approx.end());
            std::sort(approx.begin(), approx.begin()+keep);
            approx.resize(keep);
        }

        auto t1 = high_resolution_clock::now();
        double tApprox = duration<double>(t1 - t0).count();
        totalApproxTime += tApprox;

        // --- True exact search for metrics ---
        auto t2 = high_resolution_clock::now();
        std::vector<std::pair<double,int>> distTrue;
        distTrue.reserve(data_.vectors.size());
        for (const auto& v : data_.vectors)
            distTrue.emplace_back(l2(q, v.values), v.id);
        int topN = std::min(Nret, (int)distTrue.size());
        std::nth_element(distTrue.begin(), distTrue.begin()+topN, distTrue.end());
        std::sort(distTrue.begin(), distTrue.begin()+topN);
        distTrue.resize(topN);
        auto t3 = high_resolution_clock::now();
        double tTrue = duration<double>(t3 - t2).count();
        totalTrueTime += tTrue;

        // --- Metrics ---
        double AFq = 1.0, recallq = 0.0;
        if (!approx.empty() && !distTrue.empty()) {
            double da = approx[0].first;
            double dt = distTrue[0].first;
            AFq = (dt > 0.0) ? da / dt : 1.0;

            // recall@N
            std::unordered_set<int> trueSet;
            for (auto &p : distTrue) trueSet.insert(p.second);
            int hits = 0;
            for (auto &p : approx)
                if (trueSet.count(p.second)) ++hits;
            recallq = (double)hits / (double)Nret;
        }

        totalAF     += AFq;
        totalRecall += recallq;

        // --- Output per query ---
        out << "Query: " << qi << "\n";
        out << std::fixed << std::setprecision(6);
        for (int i = 0; i < (int)approx.size(); ++i) {
            out << "Nearest neighbor-" << (i+1) << ": " << approx[i].second << "\n";
            out << "distanceApproximate: " << approx[i].first << "\n";
            out << "distanceTrue: " << distTrue[ std::min(i, (int)distTrue.size()-1) ].first << "\n";
        }
        if (doRange && !rlist.empty()) {
            out << "R-near neighbors:\n";
            for (int id : rlist) out << id << "\n";
        }
        out << "\n";
    }

    // --- Summary ---
    out << "---- Summary (averages over queries) ----\n";
    out << std::fixed << std::setprecision(6);
    out << "Average AF: " << (totalAF / Q) << "\n";
    out << "Recall@N: " << (totalRecall / Q) << "\n";
    out << "QPS: " << ((totalApproxTime/Q) > 0 ? 1.0 / (totalApproxTime / Q) : 0.0) << "\n";
    out << "tApproximateAverage: " << (totalApproxTime / Q) << "\n";
    out << "tTrueAverage: " << (totalTrueTime / Q) << "\n";
}
