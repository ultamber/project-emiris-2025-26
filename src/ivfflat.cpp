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
    int N = P.size(), D = P[0].size();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> uni(0, N - 1);

    centroids_.assign(k, std::vector<float>(D, 0));
    std::vector<int> init;
    init.reserve(k);
    for (int i = 0; i < k; ++i)
    {
        int id = uni(rng);
        while (std::find(init.begin(), init.end(), id) != init.end())
            id = uni(rng);
        init.push_back(id);
        centroids_[i] = P[id];
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
                double d = l2(P[i], centroids_[c]);
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
            std::fill(centroids_[c].begin(), centroids_[c].end(), 0);
            cnt[c] = 0;
        }
        for (int i = 0; i < N; ++i)
        {
            int c = assign[i];
            ++cnt[c];
            for (int d = 0; d < D; ++d)
                centroids_[c][d] += P[i][d];
        }
        for (int c = 0; c < k; ++c)
        {
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

void IVFFlat::buildIndex(const Dataset &data)
{
    data_ = data;
    dim_ = data.dimension;
    k_ = args.kclusters;
    nprobe_ = args.nprobe;

    std::vector<std::vector<float>> P;
    P.reserve(data_.vectors.size());
    for (auto &v : data_.vectors)
        P.push_back(v.values);

    kmeans(P, k_, 25, (unsigned)args.seed);
}

void IVFFlat::search(const Dataset &queries, std::ofstream &out)
{
    using namespace std::chrono;
    out << "IVFFlat\n";

    double R = args.R;
    bool doRange = args.rangeSearch;

    double totalAF = 0.0, totalRecall = 0.0;
    double totalApproxTime = 0.0, totalTrueTime = 0.0;
    int queryCount = 0;

    auto startAll = high_resolution_clock::now();

    for (int qi = 0; qi < (int)queries.vectors.size(); ++qi)
    {
        const auto &q = queries.vectors[qi].values;
        auto startApprox = high_resolution_clock::now();

        // nearest centroids
        std::vector<std::pair<double, int>> cds;
        cds.reserve(k_);
        for (int c = 0; c < k_; ++c)
            cds.emplace_back(l2(q, centroids_[c]), c);
        std::sort(cds.begin(), cds.end());
        int use = std::min(nprobe_, (int)cds.size());

        // candidates
        std::vector<int> cand;
        for (int i = 0; i < use; ++i)
        {
            int cid = cds[i].second;
            cand.insert(cand.end(), lists_[cid].begin(), lists_[cid].end());
        }

        int idA = -1;
        double bestA = std::numeric_limits<double>::infinity();
        std::vector<int> rlist;

        if (!cand.empty())
        {
            for (int id : cand)
            {
                double d = l2(q, data_.vectors[id].values);
                if (doRange && d <= R)
                    rlist.push_back(id);
                if (d < bestA)
                {
                    bestA = d;
                    idA = id;
                }
            }
        }

        auto endApprox = high_resolution_clock::now();
        double tApprox = duration<double>(endApprox - startApprox).count();

        // compute true nearest
        auto startTrue = high_resolution_clock::now();
        int idT = -1;
        double bestT = std::numeric_limits<double>::infinity();
        for (int id = 0; id < (int)data_.vectors.size(); ++id)
        {
            double d = l2(q, data_.vectors[id].values);
            if (d < bestT)
            {
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

        if (idA != -1)
        {
            totalAF += AF;
            totalRecall += recall;
            queryCount++;
        }

        out << "Query: " << qi << "\n";
        if (idA >= 0)
        {
            out << "Nearest neighbor-1: " << idA << "\n"
                << "distanceApproximate: " << bestA << "\n"
                << "distanceTrue: " << bestT << "\n";
        }
        else
        {
            out << "Nearest neighbor-1: -1\n"
                << "distanceApproximate: inf\n"
                << "distanceTrue: " << bestT << "\n";
        }
        out << "R-near neighbors:\n";
        for (int id : rlist)
            out << id << "\n";
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