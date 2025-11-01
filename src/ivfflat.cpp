#include "ivfflat.hpp"
#include <random>
#include <limits>
#include <algorithm>
#include <chrono>
#include <iomanip>

double IVFFlat::l2(const std::vector<float> &a, const std::vector<float> &b)
{
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

void IVFFlat::kmeans(const std::vector<std::vector<float>> &P, int k, int iters, unsigned seed)
{
    int N = (int)P.size(), D = (int)P[0].size();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> uni(0, N - 1);

    // --- init: distinct random centers (simple Lloyd’s) ---
    centroids_.assign(k, std::vector<float>(D, 0.0f));
    std::vector<int> init; init.reserve(k);
    while ((int)init.size() < k) {
        int id = uni(rng);
        if (std::find(init.begin(), init.end(), id) == init.end()) {
            init.push_back(id);
            centroids_[(int)init.size()-1] = P[id];
        }
    }

    std::vector<int> assign(N, -1), cnt(k, 0);

    for (int it = 0; it < iters; ++it)
    {
        bool changed = false;

        // Assignment
        for (int i = 0; i < N; ++i) {
            int best = -1; double bd = std::numeric_limits<double>::infinity();
            for (int c = 0; c < k; ++c) {
                double d = l2(P[i], centroids_[c]);
                if (d < bd) { bd = d; best = c; }
            }
            if (assign[i] != best) { assign[i] = best; changed = true; }
        }
        if (!changed) break;

        // Update
        for (int c = 0; c < k; ++c) {
            std::fill(centroids_[c].begin(), centroids_[c].end(), 0.0f);
            cnt[c] = 0;
        }
        for (int i = 0; i < N; ++i) {
            int c = assign[i];
            ++cnt[c];
            for (int d = 0; d < D; ++d) centroids_[c][d] += P[i][d];
        }
        for (int c = 0; c < k; ++c) {
            if (cnt[c]) for (int d = 0; d < D; ++d) centroids_[c][d] /= (float)cnt[c];
            else centroids_[c] = P[uni(rng)]; // re-seed empty cluster
        }
    }
}

void IVFFlat::buildIndex(const Dataset &data)
{
    data_ = data;
    dim_ = data.dimension;
    k_ = args.kclusters;          // number of coarse cells
    nprobe_ = args.nprobe;        // #cells to probe at query

    const int N = (int)data_.vectors.size();
    if (N == 0 || k_ <= 0) { centroids_.clear(); lists_.clear(); return; }

    // --- 1) Make a √n training subset X' for centroids (slides) ---
    int trainN = std::max(k_, (int)std::sqrt((double)N));
    std::mt19937_64 rng(args.seed);
    std::vector<int> idx(N); std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(trainN);

    std::vector<std::vector<float>> Ptrain;
    Ptrain.reserve(trainN);
    for (int id : idx) Ptrain.push_back(data_.vectors[id].values);

    // --- 2) Learn centroids on X' using Lloyd’s ---
    kmeans(Ptrain, k_, /*iters*/25, (unsigned)args.seed);

    // --- 3) Build inverted lists by assigning ALL points to their nearest centroid ---
    lists_.assign(k_, {});
    for (int i = 0; i < N; ++i) {
        const auto &x = data_.vectors[i].values;
        int best = 0;
        double bd = l2(x, centroids_[0]);
        for (int c = 1; c < k_; ++c) {
            double d = l2(x, centroids_[c]);
            if (d < bd) { bd = d; best = c; }
        }
        lists_[best].push_back(i);
    }
}


void IVFFlat::search(const Dataset &queries, std::ofstream &out)
{
    using namespace std::chrono;
    out << "IVFFlat\n\n";

    const bool doRange = args.rangeSearch;
    const double Rrad = args.R;           // only for optional radius results
    const int Nret = std::max(1, args.N); // how many neighbors to return

    double totalAF=0.0, totalRecall=0.0, totalApprox=0.0, totalTrue=0.0;
    int counted=0;

    for (int qi = 0; qi < (int)queries.vectors.size(); ++qi)
    {
        const auto &q = queries.vectors[qi].values;

        // --- Approximate phase: coarse search + candidate scan ---
        auto t0 = high_resolution_clock::now();

        // 1) Score all centroids; pick top nprobe_
        std::vector<std::pair<double,int>> cds; cds.reserve(k_);
        for (int c = 0; c < k_; ++c) cds.emplace_back(l2(q, centroids_[c]), c);
        std::nth_element(cds.begin(),
                         cds.begin() + std::min(nprobe_, (int)cds.size()),
                         cds.end());
        std::sort(cds.begin(), cds.begin() + std::min(nprobe_, (int)cds.size()));
        int use = std::min(nprobe_, (int)cds.size());

        // 2) Merge candidates from the selected inverted lists
        std::vector<int> cand;
        size_t totalIn = 0;
        for (int i = 0; i < use; ++i) totalIn += lists_[cds[i].second].size();
        cand.reserve(totalIn);
        for (int i = 0; i < use; ++i) {
            int cid = cds[i].second;
            const auto &L = lists_[cid];
            cand.insert(cand.end(), L.begin(), L.end());
        }

        // 3) Compute distances only for U = ⋃ selected lists
        std::vector<std::pair<double,int>> distApprox;
        distApprox.reserve(cand.size());
        std::vector<int> rlist;
        for (int id : cand) {
            double d = l2(q, data_.vectors[id].values);
            distApprox.emplace_back(d, id);
            if (doRange && d <= Rrad) rlist.push_back(id);
        }

        // 4) Keep top-Nret approximate
        int keepA = std::min(Nret, (int)distApprox.size());
        if (keepA > 0) {
            std::nth_element(distApprox.begin(), distApprox.begin()+keepA, distApprox.end());
            std::sort(distApprox.begin(), distApprox.begin()+keepA);
            distApprox.resize(keepA);
        }

        double tApprox = duration<double>(high_resolution_clock::now() - t0).count();
        totalApprox += tApprox;

        // --- True phase: compute exact top-Nret over full DB for metrics ---
        auto t2 = high_resolution_clock::now();
        std::vector<std::pair<double,int>> distTrue;
        distTrue.reserve(data_.vectors.size());
        for (const auto &v : data_.vectors)
            distTrue.emplace_back(l2(q, v.values), v.id);
        int keepT = std::min(Nret, (int)distTrue.size());
        if (keepT > 0) {
            std::nth_element(distTrue.begin(), distTrue.begin()+keepT, distTrue.end());
            std::sort(distTrue.begin(), distTrue.begin()+keepT);
            distTrue.resize(keepT);
        }
        double tTrue = duration<double>(high_resolution_clock::now() - t2).count();
        totalTrue += tTrue;

        // --- Quality metrics (AF, Recall@N) ---
        double AFq = 0.0, Rq = 0.0;
        if (keepA > 0 && keepT > 0) {
            for (int i = 0; i < keepA; ++i) {
                double da = distApprox[i].first;
                double dt = distTrue[i].first;
                AFq += (dt > 0.0 ? da / dt : 1.0);
                int aid = distApprox[i].second;
                for (int j = 0; j < keepT; ++j)
                    if (aid == distTrue[j].second) { Rq += 1.0; break; }
            }
            AFq /= keepA;
            Rq  /= keepT;
            totalAF += AFq;
            totalRecall += Rq;
            ++counted;
        }

        // --- Output per query (top-N + radius list) ---
        out << "Query: " << qi << "\n";
        out << std::fixed << std::setprecision(6);
        for (int i = 0; i < keepA; ++i) {
            out << "Nearest neighbor-" << (i+1) << ": " << distApprox[i].second << "\n";
            out << "distanceApproximate: " << distApprox[i].first << "\n";
            out << "distanceTrue: " << distTrue[std::min(i, keepT-1)].first << "\n";
        }
        out << "R-near neighbors:\n";
        for (int id : rlist) out << id << "\n";
        out << "\n";
    }

    // --- Summary (averages over processed queries) ---
    out << std::fixed << std::setprecision(6);
    double avgAF     = (counted > 0) ? totalAF     / counted : 0.0;
    double avgRecall = (counted > 0) ? totalRecall / counted : 0.0;
    double avgApprox = (counted > 0) ? totalApprox / counted : 0.0;
    double avgTrue   = (counted > 0) ? totalTrue   / counted : 0.0;
    double qps       = (avgApprox > 0.0) ? 1.0 / avgApprox : 0.0;

    out << "Average AF: " << avgAF << "\n";
    out << "Recall@N: " << avgRecall << "\n";
    out << "QPS: " << qps << "\n";
    out << "tApproximateAverage: " << avgApprox << "\n";
    out << "tTrueAverage: " << avgTrue << "\n";
}


// --- Overall silhouette score (as before) ---
double IVFFlat::silhouetteScore() const
{
    int N = data_.vectors.size();
    int k = centroids_.size();
    if (N == 0 || k <= 1)
        return 0.0;

    std::vector<int> label(N, -1);
    for (int c = 0; c < k; ++c)
        for (int id : lists_[c])
            label[id] = c;

    double total = 0.0;
    int validPoints = 0;

    for (int i = 0; i < N; ++i)
    {
        int ci = label[i];
        if (ci < 0)
            continue;
        const auto &xi = data_.vectors[i].values;

        // a(i): mean distance within same cluster
        double a_i = 0.0;
        int sameCount = 0;
        for (int id : lists_[ci])
        {
            if (id == i)
                continue;
            a_i += l2(xi, data_.vectors[id].values);
            ++sameCount;
        }
        if (sameCount > 0)
            a_i /= sameCount;
        else
            a_i = 0.0;

        // b(i): mean distance to nearest other cluster
        double b_i = std::numeric_limits<double>::infinity();
        for (int c = 0; c < k; ++c)
        {
            if (c == ci || lists_[c].empty())
                continue;
            double sum = 0.0;
            for (int id : lists_[c])
                sum += l2(xi, data_.vectors[id].values);
            double avg = sum / lists_[c].size();
            if (avg < b_i)
                b_i = avg;
        }

        // silhouette formula from the lecture
        double s_i = 0.0;
        if (a_i < b_i)
            s_i = 1.0 - (a_i / b_i);
        else if (a_i > b_i)
            s_i = (b_i / a_i) - 1.0;
        else
            s_i = 0.0;

        total += s_i;
        ++validPoints;
    }

    return (validPoints > 0) ? total / validPoints : 0.0;
}

// --- Per-cluster silhouette averages (Slide 45) ---
std::vector<double> IVFFlat::silhouettePerCluster() const
{
    int N = data_.vectors.size();
    int k = centroids_.size();
    if (N == 0 || k <= 1)
        return {};

    std::vector<int> label(N, -1);
    for (int c = 0; c < k; ++c)
        for (int id : lists_[c])
            label[id] = c;

    std::vector<double> clusterSum(k, 0.0);
    std::vector<int> clusterCount(k, 0);

    for (int i = 0; i < N; ++i)
    {
        int ci = label[i];
        if (ci < 0)
            continue;
        const auto &xi = data_.vectors[i].values;

        // a(i)
        double a_i = 0.0;
        int sameCount = 0;
        for (int id : lists_[ci])
        {
            if (id == i)
                continue;
            a_i += l2(xi, data_.vectors[id].values);
            ++sameCount;
        }
        if (sameCount > 0)
            a_i /= sameCount;
        else
            a_i = 0.0;

        // b(i)
        double b_i = std::numeric_limits<double>::infinity();
        for (int c = 0; c < k; ++c)
        {
            if (c == ci || lists_[c].empty())
                continue;
            double sum = 0.0;
            for (int id : lists_[c])
                sum += l2(xi, data_.vectors[id].values);
            double avg = sum / lists_[c].size();
            if (avg < b_i)
                b_i = avg;
        }

        double s_i = 0.0;
        if (a_i < b_i)
            s_i = 1.0 - (a_i / b_i);
        else if (a_i > b_i)
            s_i = (b_i / a_i) - 1.0;
        else
            s_i = 0.0;

        clusterSum[ci] += s_i;
        ++clusterCount[ci];
    }

    std::vector<double> clusterAvg(k, 0.0);
    for (int c = 0; c < k; ++c)
        if (clusterCount[c] > 0)
            clusterAvg[c] = clusterSum[c] / clusterCount[c];
        else
            clusterAvg[c] = 0.0;

    return clusterAvg;
}