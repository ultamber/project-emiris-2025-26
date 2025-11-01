#include "lsh.hpp"
#include <random>
#include <limits>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>

/**
 * Computes the L2 (Euclidean) distance between two vectors
 * @param a First vector
 * @param b Second vector
 * @return The Euclidean distance between vectors a and b
 */
double LSH::l2(const std::vector<float> &a, const std::vector<float> &b)
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
 * Builds the LSH index for the input dataset
 * @param data Input dataset to be indexed
 */
void LSH::buildIndex(const Dataset &data)
{
    data_ = data;
    dim_  = data.dimension;
    w_    = args.w > 0 ? args.w : 4.0f;
    int L = args.L > 0 ? args.L : 10;
    int k = args.k > 0 ? args.k : 4;

    // 1️⃣ TableSize heuristic (slide 20)
    tableSize_ = std::max<size_t>(1, data_.vectors.size() / 4);

    // 2️⃣ Random setup
    std::mt19937_64 rng(args.seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<float> unif(0.0f, w_);
    std::uniform_int_distribution<uint32_t> distR(1u, (uint32_t)(MOD_M - 1));

    // 3️⃣ Allocate parameters
    a_.assign(L, std::vector<std::vector<float>>(k, std::vector<float>(dim_)));
    t_.assign(L, std::vector<float>(k, 0.0f));
    r_.assign(L, std::vector<long long>(k));

    // 4️⃣ Generate random vectors, shifts, and integer coefficients
    for (int li = 0; li < L; ++li) {
        for (int j = 0; j < k; ++j) {
            for (int d = 0; d < dim_; ++d)
                a_[li][j][d] = (float)normal(rng);
            t_[li][j] = unif(rng);
            r_[li][j] = (long long)distR(rng);
        }
    }

    // 5️⃣ Allocate L tables of TableSize buckets
    tables_.assign(L, std::vector<std::vector<std::pair<int, std::uint64_t>>>(tableSize_));

    // 6️⃣ Insert data points
    for (int id = 0; id < (int)data_.vectors.size(); ++id) {
        const auto &p = data_.vectors[id].values;
        for (int li = 0; li < L; ++li) {
            std::uint64_t IDp = computeID(p, li);
            std::uint64_t g   = IDp % tableSize_;
            tables_[li][g].push_back({id, IDp});
        }
    }

    if (std::getenv("LSH_DEBUG")) {
        std::cerr << "[LSH-DIAG] w=" << w_
                  << " L=" << L << " k=" << k
                  << " TableSize=" << tableSize_
                  << " min_h_seen=" << (min_h_seen_==LLONG_MAX?0:min_h_seen_)
                  << " neg_h_count=" << neg_h_count_
                  << "\n";
    }
}


/**
 * Computes the hash key for a vector in a specific hash table
 * @param v Input vector to be hashed
 * @param li Index of the hash table
 * @return 64-bit hash key
 */
std::uint64_t LSH::keyFor(const std::vector<float> &v, int li) const
{
    std::uint64_t IDv = computeID(v, li);
    return IDv % tableSize_; // g(p) = ID(p) mod TableSize
}

/**
 * Performs LSH search for all queries
 * @param queries Query dataset
 * @param out Output file stream for results
 */
void LSH::search(const Dataset &queries, std::ofstream &out)
{
    using namespace std::chrono;
    out << "LSH\n\n";

    double totalAF=0, totalRecall=0, totalApprox=0, totalTrue=0;
    int qCount = (int)queries.vectors.size();

    for (int qi = 0; qi < qCount; ++qi) {
        const auto &q = queries.vectors[qi].values;
        auto t0 = high_resolution_clock::now();

        std::vector<int> candidates;
        size_t examined=0, hardCap = args.rangeSearch ? 20*args.L : 10*args.L;

        // 1️⃣ Probe all L tables
        for (int li = 0; li < args.L; ++li) {
            std::uint64_t IDq = computeID(q, li);
            std::uint64_t gq  = IDq % tableSize_;
            const auto &bucket = tables_[li][gq];

            for (const auto &pr : bucket) {
                if (pr.second == IDq) {
                    candidates.push_back(pr.first);
                    if (++examined > hardCap) break;
                }
            }
            if (examined > hardCap) break;
        }

        // 2️⃣ Deduplicate
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

        // 3️⃣ Distance check
        std::vector<std::pair<double,int>> distApprox;
        std::vector<int> rlist;
        distApprox.reserve(candidates.size());

        for (int id : candidates) {
            double d = l2(q, data_.vectors[id].values);
            distApprox.emplace_back(d,id);
            if (args.rangeSearch && d <= args.R) rlist.push_back(id);
        }

        // 4️⃣ Keep top N
        int N = std::min(args.N, (int)distApprox.size());
        if (N>0){
            std::nth_element(distApprox.begin(), distApprox.begin()+N, distApprox.end());
            std::sort(distApprox.begin(), distApprox.begin()+N);
        }
        double tApprox = duration<double>(high_resolution_clock::now()-t0).count();
        totalApprox += tApprox;

        // 5️⃣ True NN for evaluation
        auto t2 = high_resolution_clock::now();
        std::vector<std::pair<double,int>> distTrue;
        distTrue.reserve(data_.vectors.size());
        for (auto &v : data_.vectors)
            distTrue.emplace_back(l2(q, v.values), v.id);
        std::nth_element(distTrue.begin(), distTrue.begin()+N, distTrue.end());
        std::sort(distTrue.begin(), distTrue.begin()+N);
        double tTrue = duration<double>(high_resolution_clock::now()-t2).count();
        totalTrue += tTrue;

        // 6️⃣ Metrics
        double AFq=0, recallq=0;
        for (int i=0;i<N;++i){
            double da=distApprox[i].first, dt=distTrue[i].first;
            AFq+=(dt>0?da/dt:1.0);
            int aid=distApprox[i].second;
            for (int j=0;j<N;++j)
                if(aid==distTrue[j].second){recallq+=1;break;}
        }
        if (N>0){AFq/=N;recallq/=N;}
        totalAF+=AFq;totalRecall+=recallq;

        // 7️⃣ Output
        out<<"Query: "<<qi<<"\n"<<std::fixed<<std::setprecision(6);
        for(int i=0;i<N;++i){
            out<<"Nearest neighbor-"<<(i+1)<<": "<<distApprox[i].second<<"\n";
            out<<"distanceApproximate: "<<distApprox[i].first<<"\n";
            out<<"distanceTrue: "<<distTrue[i].first<<"\n";
        }
        out<<"\nR-near neighbors:\n";
        for(int id:rlist) out<<id<<"\n";
        out<<"\n";
    }

    // 8️⃣ Summary
    out<<"---- Summary (averages over queries) ----\n";
    out<<"Average AF: "<<(totalAF/qCount)<<"\n";
    out<<"Recall@N: "<<(totalRecall/qCount)<<"\n";
    out<<"QPS: "<<(1.0/(totalApprox/qCount))<<"\n";
    out<<"tApproximateAverage: "<<(totalApprox/qCount)<<"\n";
    out<<"tTrueAverage: "<<(totalTrue/qCount)<<"\n";
}
