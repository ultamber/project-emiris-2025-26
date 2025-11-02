#include "ivfpq.hpp"
#include <random>
#include <limits>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>
#include <unordered_set>
#include <iostream>

/**
 * Computes the L2 (Euclidean) distance between two vectors
 */
double IVFPQ::l2(const std::vector<float> &a, const std::vector<float> &b) const
{
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double d = a[i] - b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

/**
 * Lloyd's algorithm for k-means clustering ref 31)
 */
void IVFPQ::kmeans(const std::vector<std::vector<float>> &P, int k, int iters, unsigned seed,
                   std::vector<std::vector<float>> &C, std::vector<int> *outAssign)
{
    int N = (int)P.size(), D = (int)P[0].size();
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> uni(0, N - 1);

    // Initialize k distinct random centers
    C.assign(k, std::vector<float>(D, 0.0f));
    std::vector<int> init;
    init.reserve(k);
    while ((int)init.size() < k)
    {
        int id = uni(rng);
        if (std::find(init.begin(), init.end(), id) == init.end())
        {
            init.push_back(id);
            C[(int)init.size() - 1] = P[id];
        }
    }

    std::vector<int> assign(N, -1);
    std::vector<int> cnt(k, 0);

    // Lloyd's iterations
    for (int it = 0; it < iters; ++it)
    {
        bool changed = false;

        // Assignment step
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
                changed = true;
            }
        }

        if (!changed)
            break;

        // Update step
        for (int c = 0; c < k; ++c)
        {
            std::fill(C[c].begin(), C[c].end(), 0.0f);
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
            {
                for (int d = 0; d < D; ++d)
                    C[c][d] /= (float)cnt[c];
            }
            else
            {
                C[c] = P[uni(rng)]; // Re-seed empty cluster
            }
        }
    }

    if (outAssign)
        *outAssign = std::move(assign);
}

/**
 * Computes residual vectors r(x) = x - c(x) for all points ref 49
 */
void IVFPQ::makeResiduals(const Dataset &data, const std::vector<int> &assign,
                          std::vector<std::vector<float>> &residuals)
{
    int N = (int)data.vectors.size(), D = data.dimension;
    residuals.assign(N, std::vector<float>(D, 0.0f));
    for (int i = 0; i < N; ++i)
    {
        int c = assign[i];
        for (int d = 0; d < D; ++d)
            residuals[i][d] = data.vectors[i].values[d] - centroids_[c][d];
    }
}

/**
 * Trains Product Quantizer on residuals ref 49
 * Split residual into M subspaces, run k-means on each with s=2^nbits centroids
 */
void IVFPQ::trainPQ(const std::vector<std::vector<float>> &R)
{
    int D = dim_, M = M_;
    
    // slide 49Split r(x) into M parts
    subdim_.assign(M, D / M);
    for (int i = 0; i < D % M; ++i)
        ++subdim_[i]; // Distribute remainder

    codebooks_.assign(M, {});
    unsigned seed = (unsigned)args.seed;

    int offset = 0;
    for (int m = 0; m < M; ++m)
    {
        int sd = subdim_[m];
        
        // Extract subspace m from all residuals
        std::vector<std::vector<float>> subspace;
        subspace.reserve(R.size());
        for (const auto &r : R)
            subspace.emplace_back(r.begin() + offset, r.begin() + offset + sd);

        // slide 49 Lloyd's creates clustering with s = 2^nbits centroids
        std::vector<std::vector<float>> cb;
        kmeans(subspace, Ks_, 20, seed + m, cb, nullptr);
        kmeansWithPP(subspace, k_, 20, (unsigned)args.seed + m, centroids_, nullptr);
        codebooks_[m] = std::move(cb);
        
        offset += sd;
    }
}

/**
 * Encodes all residuals using trained PQ ref 49
 */
void IVFPQ::encodeAll(const std::vector<std::vector<float>> &R)
{
    int N = (int)R.size(), M = M_;
    codes_.assign(N, std::vector<uint8_t>(M, 0));

    int offset = 0;
    for (int m = 0; m < M; ++m)
    {
        int sd = subdim_[m];
        
        // slide 49 For each subspace, find nearest centroid
        for (int i = 0; i < N; ++i)
        {
            int best = 0;
            double bd = std::numeric_limits<double>::infinity();
            
            for (int k = 0; k < Ks_; ++k)
            {
                double s = 0.0;
                for (int d = 0; d < sd; ++d)
                {
                    double diff = R[i][offset + d] - codebooks_[m][k][d];
                    s += diff * diff;
                }
                if (s < bd)
                {
                    bd = s;
                    best = k;
                }
            }
            
            // slide 49 code_i(x) = argmin_h ||r_i(x) - c_{i,h}||_2
            codes_[i][m] = (uint8_t)best;
        }
        offset += sd;
    }
}

/**
 * Builds the IVFPQ index ref 49
 */
void IVFPQ::buildIndex(const Dataset &data)
{
    data_ = data;
    dim_ = data.dimension;

    k_ = args.kclusters;         // Number of coarse cells (k in slides)
    nprobe_ = args.nprobe;       // Number of cells to probe (b in slides)
    M_ = args.Msubvectors;       // Number of subquantizers (M in slides)
    Ks_ = 1 << args.nbits;       // Codebook size per subquantizer (s = 2^nbits in slides)

    const int N = (int)data_.vectors.size();
    if (N == 0)
    {
        centroids_.clear();
        lists_.clear();
        return;
    }

    // slide 49, Run Lloyd's on subset X' ~ √n to obtain centroids {c_j}
    int trainN = std::max(k_, (int)std::sqrt((double)N));
    trainN = std::min(trainN, N);

    std::mt19937_64 rng(args.seed);
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    idx.resize(trainN);

    std::vector<std::vector<float>> Ptrain;
    Ptrain.reserve(trainN);
    for (int id : idx)
        Ptrain.push_back(data_.vectors[id].values);

    kmeans(Ptrain, k_, 40, (unsigned)args.seed, centroids_, nullptr);

    // slide 49, Assign ALL points to nearest centroid
    lists_.assign(k_, {});
    std::vector<int> fullAssign(N, -1);
    for (int i = 0; i < N; ++i)
    {
        const auto &x = data_.vectors[i].values;
        int best = 0;
        double bd = l2(x, centroids_[0]);
        for (int c = 1; c < k_; ++c)
        {
            double d = l2(x, centroids_[c]);
            if (d < bd)
            {
                bd = d;
                best = c;
            }
        }
        lists_[best].push_back(i);
        fullAssign[i] = best;
    }

    // slide 49,  Compute residual vectors r(x) = x - c(x)
    std::vector<std::vector<float>> R;
    makeResiduals(data_, fullAssign, R);

    // slide 49, Train product quantizer on residuals
    trainPQ(R);

    // slide 49,  Encode ALL residuals with trained PQ
    // PQ(x) = [code_1(x), ..., code_M(x)]
    encodeAll(R);
}

/**
 * Performs IVFPQ search ref 50
 */
void IVFPQ::search(const Dataset &queries, std::ofstream &out)
{
    using namespace std::chrono;
    out << "IVFPQ\n\n";

    const double Rrad = args.R;
    const bool doRange = args.rangeSearch;
    const int Nret = std::max(1, args.N);
    const int rerankTop = 100; // Re-rank top T candidates with exact distances

    double totalAF = 0.0, totalRecall = 0.0;
    double totalApproxTime = 0.0, totalTrueTime = 0.0;
    int counted = 0;
    int Q = (int)queries.vectors.size();

    if (args.maxQueries > 0)
        Q = std::min(Q, args.maxQueries);

    for (int qi = 0; qi < Q; ++qi)
    {
        const auto &q = queries.vectors[qi].values;

        // === Approximate search using IVFPQ ===
        auto t0 = high_resolution_clock::now();

        // slide 50Score centroids - compute ||q - c_j||_2
        std::vector<std::pair<double, int>> centroidDists;
        centroidDists.reserve(k_);
        for (int c = 0; c < k_; ++c)
        {
            double s = 0.0;
            for (int d = 0; d < dim_; ++d)
            {
                double diff = q[d] - centroids_[c][d];
                s += diff * diff;
            }
            centroidDists.emplace_back(s, c); // Store squared distance
        }

        // Select top b probes
        int probeCount = std::min(nprobe_, (int)centroidDists.size());
        std::nth_element(centroidDists.begin(),
                        centroidDists.begin() + probeCount,
                        centroidDists.end());
        std::sort(centroidDists.begin(), centroidDists.begin() + probeCount);

        std::vector<std::pair<double, int>> adcCandidates; // (adc_distance_squared, id)
        std::vector<int> rlist;

        // slide 50 For each selected c_j
        for (int pi = 0; pi < probeCount; ++pi)
        {
            int cid = centroidDists[pi].second;

            // Compute residual: r(q) = q - c_j
            std::vector<float> rq(dim_);
            for (int d = 0; d < dim_; ++d)
                rq[d] = q[d] - centroids_[cid][d];

            // Split r(q) = [r_1(q), ..., r_M(q)]
            // Define LUT[i][h] = ||r_i(q) - c_{i,h}||_2 ref 50)
            std::vector<std::vector<double>> LUT(M_, std::vector<double>(Ks_, 0.0));
            int offset = 0;
            for (int m = 0; m < M_; ++m)
            {
                int sd = subdim_[m];
                for (int k = 0; k < Ks_; ++k)
                {
                    double s = 0.0;
                    for (int d = 0; d < sd; ++d)
                    {
                        double diff = rq[offset + d] - codebooks_[m][k][d];
                        s += diff * diff;
                    }
                    LUT[m][k] = s; // Squared distance
                }
                offset += sd;
            }

            // slide 50 Asymmetric Distance Computation (ADC)
            // For x in U, d(q,x) = Σ_i LUT[i][code_i(x)]
            for (int id : lists_[cid])
            {
                double adc_sqr = 0.0;
                for (int m = 0; m < M_; ++m)
                    adc_sqr += LUT[m][codes_[id][m]];
                
                adcCandidates.emplace_back(adc_sqr, id);

                // Range search in PQ space (optional)
                if (doRange)
                {
                    double adc = std::sqrt(adc_sqr);
                    if (adc <= Rrad)
                        rlist.push_back(id);
                }
            }
        }

        // Keep top T candidates for re-ranking
        std::vector<std::pair<double, int>> reranked;
        if (!adcCandidates.empty())
        {
            int T = std::max(Nret, rerankTop);
            T = std::min(T, (int)adcCandidates.size());
            std::nth_element(adcCandidates.begin(),
                           adcCandidates.begin() + T,
                           adcCandidates.end());
            
            // Re-rank top T with exact distances
            reranked.reserve(T);
            for (int i = 0; i < T; ++i)
            {
                int id = adcCandidates[i].second;
                double exactDist = l2(q, data_.vectors[id].values);
                reranked.emplace_back(exactDist, id);
            }
        }

        // slide 50,Return R nearest points
        int keepApprox = std::min(Nret, (int)reranked.size());
        if (keepApprox > 0)
        {
            std::nth_element(reranked.begin(),
                           reranked.begin() + keepApprox,
                           reranked.end());
            std::sort(reranked.begin(), reranked.begin() + keepApprox);
            reranked.resize(keepApprox);
        }

        double tApprox = duration<double>(high_resolution_clock::now() - t0).count();
        totalApproxTime += tApprox;

        // === Ground truth for evaluation ===
        auto t2 = high_resolution_clock::now();
        std::vector<std::pair<double, int>> distTrue;
        distTrue.reserve(data_.vectors.size());
        for (const auto &v : data_.vectors)
            distTrue.emplace_back(l2(q, v.values), v.id);

        int keepTrue = std::min(Nret, (int)distTrue.size());
        if (keepTrue > 0)
        {
            std::nth_element(distTrue.begin(),
                           distTrue.begin() + keepTrue,
                           distTrue.end());
            std::sort(distTrue.begin(), distTrue.begin() + keepTrue);
            distTrue.resize(keepTrue);
        }

        double tTrue = duration<double>(high_resolution_clock::now() - t2).count();
        totalTrueTime += tTrue;

        // === Quality metrics ===
        double AFq = 0.0, Rq = 0.0;
        if (keepApprox > 0 && keepTrue > 0)
        {
            // Approximation Factor
            for (int i = 0; i < keepApprox; ++i)
            {
                double da = reranked[i].first;
                double dt = distTrue[i].first;
                AFq += (dt > 0.0 ? da / dt : 1.0);
            }
            AFq /= keepApprox;

            // Recall@N
            std::unordered_set<int> trueSet;
            for (const auto &p : distTrue)
                trueSet.insert(p.second);
            int hits = 0;
            for (const auto &p : reranked)
                if (trueSet.count(p.second))
                    ++hits;
            Rq = (double)hits / (double)keepTrue;

            totalAF += AFq;
            totalRecall += Rq;
            ++counted;
        }

        // === Output per query ===
        out << "Query: " << qi << "\n";
        out << std::fixed << std::setprecision(6);
        for (int i = 0; i < keepApprox; ++i)
        {
            out << "Nearest neighbor-" << (i + 1) << ": " << reranked[i].second << "\n";
            out << "distanceApproximate: " << reranked[i].first << "\n";
            out << "distanceTrue: " << distTrue[std::min(i, keepTrue - 1)].first << "\n";
        }
        out << "\nR-near neighbors:\n";
        for (int id : rlist)
            out << id << "\n";
        out << "\n";
    }

    // === Summary statistics ===
    out << "---- Summary (averages over queries) ----\n";
    out << std::fixed << std::setprecision(6);
    double avgAF = (counted > 0) ? totalAF / counted : 0.0;
    double avgRecall = (counted > 0) ? totalRecall / counted : 0.0;
    double avgApprox = (counted > 0) ? totalApproxTime / counted : 0.0;
    double avgTrue = (counted > 0) ? totalTrueTime / counted : 0.0;
    double qps = (avgApprox > 0.0) ? 1.0 / avgApprox : 0.0;

    out << "Average AF: " << avgAF << "\n";
    out << "Recall@N: " << avgRecall << "\n";
    out << "QPS: " << qps << "\n";
    out << "tApproximateAverage: " << avgApprox << "\n";
    out << "tTrueAverage: " << avgTrue << "\n";
}

/**
 * Overall silhouette score (slides 44-45)
 */
double IVFPQ::silhouetteScore() const
{
    int N = (int)data_.vectors.size();
    int k = (int)centroids_.size();
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

        // a(i): average distance within same cluster
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

        // b(i): average distance to next best cluster
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

        // s(i) = (b(i) - a(i)) / max{a(i), b(i)}
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

/**
 * Per-cluster silhouette averages ref 45)
 */
std::vector<double> IVFPQ::silhouettePerCluster() const
{
    int N = (int)data_.vectors.size();
    int k = (int)centroids_.size();
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

void IVFPQ::kmeanspp(const std::vector<std::vector<float>> &P, int k, 
                  std::vector<std::vector<float>> &centroids, unsigned seed)
    {
        int N = (int)P.size();
        // int D = (int)P[0].size();
        
        if (N == 0 || k <= 0) {
            centroids.clear();
            return;
        }
        
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> uni(0, N - 1);
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        
        centroids.clear();
        centroids.reserve(k);
        
        //Choose first centroid uniformly at random
        int firstIdx = uni(rng);
        centroids.push_back(P[firstIdx]);
        
        std::vector<double> minDist(N, std::numeric_limits<double>::infinity());
        
        //Choose remaining k-1 centroids
        for (int t = 1; t < k; ++t)
        {
            // Update D(i) = min distance to any chosen centroid
            const auto &lastCentroid = centroids.back();
            for (int i = 0; i < N; ++i)
            {
                double d = l2(P[i], lastCentroid);
                if (d < minDist[i])
                    minDist[i] = d;
            }
            
            // Normalize D(i) to avoid overflow ref 42)
            double maxD = *std::max_element(minDist.begin(), minDist.end());
            if (maxD <= 0.0) {
                // All points are centroids, choose random
                centroids.push_back(P[uni(rng)]);
                continue;
            }
            
            // Build partial sums: P(r) = Σ(i=1 to r) D(i)^2
            std::vector<double> partialSums(N, 0.0);
            partialSums[0] = (minDist[0] / maxD) * (minDist[0] / maxD);
            for (int i = 1; i < N; ++i)
            {
                double normDist = minDist[i] / maxD;
                partialSums[i] = partialSums[i-1] + normDist * normDist;
            }
            
            // Choose r with probability ∝ D(r)^2 ref 42)
            // Pick random x ∈ [0, P(N-1)] and find r where P(r-1) < x ≤ P(r)
            double x = unif(rng) * partialSums[N-1];
            
            // Binary search for r
            int r = std::lower_bound(partialSums.begin(), partialSums.end(), x) 
                    - partialSums.begin();
            r = std::min(r, N - 1);
            
            centroids.push_back(P[r]);
        }
    }
    
    /**
     * Lloyd's algorithm with k-means++ initialization
     */
    void IVFPQ::kmeansWithPP(const std::vector<std::vector<float>> &P, int k, 
                      int iters, unsigned seed,
                      std::vector<std::vector<float>> &centroids,
                      std::vector<int> *outAssign)
    {
        int N = (int)P.size();
        int D = (int)P[0].size();
        
        if (N == 0 || k <= 0) {
            centroids.clear();
            if (outAssign) outAssign->clear();
            return;
        }
        
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> uni(0, N - 1);
        
        // Use k-means++ for initialization instead of random
        kmeanspp(P, k, centroids, seed);
        
        std::vector<int> assign(N, -1);
        std::vector<int> cnt(k, 0);
        
        // Lloyd's iterations
        for (int it = 0; it < iters; ++it)
        {
            bool changed = false;
            
            // Assignment step
            for (int i = 0; i < N; ++i)
            {
                int best = -1;
                double bd = std::numeric_limits<double>::infinity();
                for (int c = 0; c < k; ++c)
                {
                    double d = l2(P[i], centroids[c]);
                    if (d < bd)
                    {
                        bd = d;
                        best = c;
                    }
                }
                if (assign[i] != best)
                {
                    assign[i] = best;
                    changed = true;
                }
            }
            
            if (!changed) break;
            
            // Update step
            for (int c = 0; c < k; ++c)
            {
                std::fill(centroids[c].begin(), centroids[c].end(), 0.0f);
                cnt[c] = 0;
            }
            for (int i = 0; i < N; ++i)
            {
                int c = assign[i];
                ++cnt[c];
                for (int d = 0; d < D; ++d)
                    centroids[c][d] += P[i][d];
            }
            for (int c = 0; c < k; ++c)
            {
                if (cnt[c])
                {
                    for (int d = 0; d < D; ++d)
                        centroids[c][d] /= (float)cnt[c];
                }
                else
                {
                    // Re-seed empty cluster
                    centroids[c] = P[uni(rng)];
                }
            }
        }
        
        if (outAssign)
            *outAssign = std::move(assign);
    }
    

    double IVFPQ::computeSilhouetteForK(int k, unsigned seed)
    {
        if (k <= 1 || k >= (int)data_.vectors.size())
            return -1.0; // Invalid k
        
        // Build temporary clustering with k clusters
        std::vector<std::vector<float>> P;
        P.reserve(data_.vectors.size());
        for (const auto &v : data_.vectors)
            P.push_back(v.values);
        
        std::vector<std::vector<float>> tempCentroids;
        std::vector<int> assign;
        kmeansWithPP(P, k, 25, seed, tempCentroids, &assign);
        
        // Build temporary inverted lists
        std::vector<std::vector<int>> tempLists(k);
        for (int i = 0; i < (int)assign.size(); ++i)
        {
            if (assign[i] >= 0 && assign[i] < k)
                tempLists[assign[i]].push_back(i);
        }
        
        // Compute silhouette score ref 44
        double total = 0.0;
        int validPoints = 0;
        int N = (int)data_.vectors.size();
        
        for (int i = 0; i < N; ++i)
        {
            int ci = assign[i];
            if (ci < 0 || ci >= k) continue;
            
            const auto &xi = data_.vectors[i].values;
            
            // a(i): average distance within same cluster
            double a_i = 0.0;
            int sameCount = 0;
            for (int id : tempLists[ci])
            {
                if (id == i) continue;
                a_i += l2(xi, data_.vectors[id].values);
                ++sameCount;
            }
            if (sameCount > 0)
                a_i /= sameCount;
            else
                a_i = 0.0;
            
            // b(i): average distance to next best cluster
            double b_i = std::numeric_limits<double>::infinity();
            for (int c = 0; c < k; ++c)
            {
                if (c == ci || tempLists[c].empty())
                    continue;
                double sum = 0.0;
                for (int id : tempLists[c])
                    sum += l2(xi, data_.vectors[id].values);
                double avg = sum / tempLists[c].size();
                if (avg < b_i)
                    b_i = avg;
            }
            
            // s(i) = (b(i) - a(i)) / max{a(i), b(i)}
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
    
    int IVFPQ::findOptimalK(int kmin, int kmax, unsigned seed, int step)
    {
        if (kmin >= kmax || kmin < 2)
        {
            std::cerr << "Invalid k range: [" << kmin << ", " << kmax << "]\n";
            return kmin;
        }
        
        std::cout << "\n=== Finding Optimal K using Silhouette Method ===\n";
        std::cout << "Testing k in range [" << kmin << ", " << kmax << "] with step " << step << "\n\n";
        
        int bestK = kmin;
        double bestScore = -2.0; // Silhouette scores are in [-1, 1]
        
        for (int k = kmin; k <= kmax; k += step)
        {
            double score = computeSilhouetteForK(k, seed);
            
            std::cout << "k = " << k 
                      << " | Silhouette = " << std::fixed << std::setprecision(4) << score;
            
            if (score > bestScore)
            {
                bestScore = score;
                bestK = k;
                std::cout << " <- NEW BEST";
            }
            std::cout << "\n";
        }
        
        std::cout << "\n=== Optimal K = " << bestK 
                  << " with Silhouette = " << bestScore << " ===\n\n";
        
        return bestK;
    }

    int IVFPQ::findOptimalKEnhanced(int kmin, int kmax, unsigned seed, int step)
    {
        std::cout << "\n=== Enhanced K Selection (Overall + Per-Cluster) ===\n";
        
        struct KScore {
            int k;
            double overallScore;
            double clusterVariance; // Variance of per-cluster scores
            double combinedScore;   // Overall - penalty*variance
        };
        
        std::vector<KScore> scores;
        
        for (int k = kmin; k <= kmax; k += step)
        {
            // Build clustering
            std::vector<std::vector<float>> P;
            P.reserve(data_.vectors.size());
            for (const auto &v : data_.vectors)
                P.push_back(v.values);
            
            std::vector<std::vector<float>> tempCentroids;
            std::vector<int> assign;
            kmeansWithPP(P, k, 25, seed, tempCentroids, &assign);
            
            std::vector<std::vector<int>> tempLists(k);
            for (int i = 0; i < (int)assign.size(); ++i)
                if (assign[i] >= 0 && assign[i] < k)
                    tempLists[assign[i]].push_back(i);
            
            // Compute per-cluster silhouettes
            std::vector<double> clusterScores;
            for (int c = 0; c < k; ++c)
            {
                if (tempLists[c].empty()) continue;
                
                double clusterSum = 0.0;
                int clusterCount = 0;
                
                for (int i : tempLists[c])
                {
                    const auto &xi = data_.vectors[i].values;
                    
                    double a_i = 0.0;
                    int sameCount = 0;
                    for (int id : tempLists[c])
                    {
                        if (id == i) continue;
                        a_i += l2(xi, data_.vectors[id].values);
                        ++sameCount;
                    }
                    if (sameCount > 0) a_i /= sameCount;
                    
                    double b_i = std::numeric_limits<double>::infinity();
                    for (int c2 = 0; c2 < k; ++c2)
                    {
                        if (c2 == c || tempLists[c2].empty()) continue;
                        double sum = 0.0;
                        for (int id : tempLists[c2])
                            sum += l2(xi, data_.vectors[id].values);
                        double avg = sum / tempLists[c2].size();
                        if (avg < b_i) b_i = avg;
                    }
                    
                    double s_i = 0.0;
                    if (a_i < b_i)
                        s_i = 1.0 - (a_i / b_i);
                    else if (a_i > b_i)
                        s_i = (b_i / a_i) - 1.0;
                    
                    clusterSum += s_i;
                    ++clusterCount;
                }
                
                if (clusterCount > 0)
                    clusterScores.push_back(clusterSum / clusterCount);
            }
            
            // Overall score
            double overall = 0.0;
            for (double s : clusterScores) overall += s;
            if (!clusterScores.empty()) overall /= clusterScores.size();
            
            // Variance of cluster scores (lower is better)
            double variance = 0.0;
            if (!clusterScores.empty())
            {
                for (double s : clusterScores)
                    variance += (s - overall) * (s - overall);
                variance /= clusterScores.size();
            }
            
            // Combined score: penalize high variance
            double combined = overall - 0.2 * variance;
            
            scores.push_back({k, overall, variance, combined});
            
            std::cout << "k = " << k 
                      << " | Overall = " << std::fixed << std::setprecision(4) << overall
                      << " | Variance = " << variance
                      << " | Combined = " << combined << "\n";
        }
        
        // Find best combined score
        auto best = std::max_element(scores.begin(), scores.end(),
            [](const KScore &a, const KScore &b) {
                return a.combinedScore < b.combinedScore;
            });
        
        std::cout << "\n=== Optimal K = " << best->k 
                  << " (Combined Score = " << best->combinedScore << ") ===\n\n";
        
        return best->k;
    }