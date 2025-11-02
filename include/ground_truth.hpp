#pragma once
#include "dataset.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>

struct GroundTruth
{
    // For each query: list of (distance, id) pairs for top N neighbors
    std::vector<std::vector<std::pair<double, int>>> trueNeighbors;
    double avgTrueTime = 0.0;

    // Compute ground truth for all queries
    void compute(const Dataset &data, const Dataset &queries, int N)
    {
        using namespace std::chrono;

        static const size_t LIMIT_QUERIES = 1000; 
        size_t numQueriesToProcess = std::min(LIMIT_QUERIES, queries.vectors.size());
        std::cout << "Computing ground truth for " << numQueriesToProcess
                  << " queries (limited by static LIMIT_QUERIES=" << LIMIT_QUERIES << ")\n";

        trueNeighbors.clear();
        trueNeighbors.reserve(queries.vectors.size());

        double totalTime = 0.0;

        for (size_t qi = 0; qi < numQueriesToProcess; ++qi)
        {
            const auto &q = queries.vectors[qi].values;

            auto t0 = high_resolution_clock::now();

            // Brute force: compute distance to ALL points
            std::vector<std::pair<double, int>> distances;
            distances.reserve(data.vectors.size());

            for (const auto &v : data.vectors)
            {
                double d = l2(q, v.values);
                distances.emplace_back(d, v.id);
            }

            // Find top N
            int topN = std::min(N, (int)distances.size());
            if (topN > 0)
            {
                std::nth_element(distances.begin(),
                                 distances.begin() + topN,
                                 distances.end());
                std::sort(distances.begin(), distances.begin() + topN);
                distances.resize(topN);
            }

            auto t1 = high_resolution_clock::now();
            totalTime += duration<double>(t1 - t0).count();

            trueNeighbors.push_back(std::move(distances));
        }

        avgTrueTime = totalTime / numQueriesToProcess;
    }

    // Save to binary file for later reuse (robust, portable)
    void save(const std::string &filename) const
    {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs)
        {
            std::cerr << "GroundTruth::save: cannot open file " << filename << " for writing\n";
            return;
        }

        // use fixed-size counts for portability
        uint64_t numQueries = static_cast<uint64_t>(trueNeighbors.size());
        ofs.write(reinterpret_cast<const char *>(&numQueries), sizeof(numQueries));
        ofs.write(reinterpret_cast<const char *>(&avgTrueTime), sizeof(avgTrueTime));

        for (const auto &neighbors : trueNeighbors)
        {
            uint64_t numNeighbors = static_cast<uint64_t>(neighbors.size());
            ofs.write(reinterpret_cast<const char *>(&numNeighbors), sizeof(numNeighbors));

            // write each pair element-by-element to avoid any padding/ABI issues
            for (const auto &p : neighbors)
            {
                double dist = p.first;
                int id = p.second;
                ofs.write(reinterpret_cast<const char *>(&dist), sizeof(dist));
                ofs.write(reinterpret_cast<const char *>(&id), sizeof(id));
            }
        }

        ofs.close();
    }

    // Load from binary file (robust checks and diagnostics)
    bool load(const std::string &filename)
    {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs)
            return false;

        // get file size (optional but useful to detect truncation)
        ifs.seekg(0, std::ios::end);
        std::streampos fileSize = ifs.tellg();
        ifs.seekg(0, std::ios::beg);

        uint64_t numQueries64 = 0;
        ifs.read(reinterpret_cast<char *>(&numQueries64), sizeof(numQueries64));
        if (!ifs)
        {
            std::cerr << "GroundTruth::load: failed reading numQueries\n";
            return false;
        }

        ifs.read(reinterpret_cast<char *>(&avgTrueTime), sizeof(avgTrueTime));
        if (!ifs)
        {
            std::cerr << "GroundTruth::load: failed reading avgTrueTime\n";
            return false;
        }

        // sanity checks
        const uint64_t MAX_QUERIES = 10000000ULL; // tune to something reasonable for your dataset
        if (numQueries64 == 0 || numQueries64 > MAX_QUERIES)
        {
            std::cerr << "GroundTruth::load: suspicious numQueries=" << numQueries64 << " (file: " << filename << ")\n";
            return false;
        }

        size_t numQueries = static_cast<size_t>(numQueries64);
        trueNeighbors.clear();
        trueNeighbors.resize(numQueries);

        // iterate and read neighbors
        for (size_t qi = 0; qi < numQueries; ++qi)
        {
            uint64_t numNeighbors64 = 0;
            ifs.read(reinterpret_cast<char *>(&numNeighbors64), sizeof(numNeighbors64));
            if (!ifs)
            {
                std::cerr << "GroundTruth::load: failed reading numNeighbors for query " << qi << "\n";
                return false;
            }

            const uint64_t MAX_NEIGHBORS = 1000000ULL; // tune to safe bound
            if (numNeighbors64 > MAX_NEIGHBORS)
            {
                std::cerr << "GroundTruth::load: suspicious numNeighbors=" << numNeighbors64 << " for query " << qi << "\n";
                return false;
            }

            size_t numNeighbors = static_cast<size_t>(numNeighbors64);
            trueNeighbors[qi].clear();
            trueNeighbors[qi].reserve(numNeighbors);

            for (size_t ni = 0; ni < numNeighbors; ++ni)
            {
                double dist;
                int id;
                ifs.read(reinterpret_cast<char *>(&dist), sizeof(dist));
                if (!ifs)
                {
                    std::cerr << "GroundTruth::load: failed reading dist for q" << qi << " n" << ni << "\n";
                    return false;
                }
                ifs.read(reinterpret_cast<char *>(&id), sizeof(id));
                if (!ifs)
                {
                    std::cerr << "GroundTruth::load: failed reading id for q" << qi << " n" << ni << "\n";
                    return false;
                }

                trueNeighbors[qi].emplace_back(dist, id);
            }
        }

        // optional: confirm we didn't unexpectedly hit EOF prematurely
        std::streampos finalPos = ifs.tellg();
        if (finalPos == -1)
            finalPos = fileSize; // some platforms return -1 after reading EOF
        if (finalPos > fileSize)
        {
            std::cerr << "GroundTruth::load: read pointer beyond file size (weird)\n";
        }

        return true;
    }

private:
    double l2(const std::vector<float> &a, const std::vector<float> &b) const
    {
        double s = 0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            double d = a[i] - b[i];
            s += d * d;
        }
        return std::sqrt(s);
    }
};