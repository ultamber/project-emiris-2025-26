#pragma once
#include "dataset.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>

struct GroundTruth {
    // For each query: list of (distance, id) pairs for top N neighbors
    std::vector<std::vector<std::pair<double, int>>> trueNeighbors;
    double avgTrueTime = 0.0;
    
    // Compute ground truth for all queries
    void compute(const Dataset &data, const Dataset &queries, int N)
    {
        using namespace std::chrono;
        
        trueNeighbors.clear();
        trueNeighbors.reserve(queries.vectors.size());
        
        double totalTime = 0.0;
        
        for (const auto &qvec : queries.vectors)
        {
            const auto &q = qvec.values;
            
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
        
        avgTrueTime = totalTime / queries.vectors.size();
    }
    
    // Save to binary file for later reuse
    void save(const std::string &filename) const
    {
        std::ofstream ofs(filename, std::ios::binary);
        
        size_t numQueries = trueNeighbors.size();
        ofs.write((char*)&numQueries, sizeof(numQueries));
        ofs.write((char*)&avgTrueTime, sizeof(avgTrueTime));
        
        for (const auto &neighbors : trueNeighbors)
        {
            size_t numNeighbors = neighbors.size();
            ofs.write((char*)&numNeighbors, sizeof(numNeighbors));
            ofs.write((char*)neighbors.data(), numNeighbors * sizeof(neighbors[0]));
        }
    }
    
    // Load from binary file
    bool load(const std::string &filename)
    {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) return false;
        
        size_t numQueries;
        ifs.read((char*)&numQueries, sizeof(numQueries));
        ifs.read((char*)&avgTrueTime, sizeof(avgTrueTime));
        
        trueNeighbors.resize(numQueries);
        for (auto &neighbors : trueNeighbors)
        {
            size_t numNeighbors;
            ifs.read((char*)&numNeighbors, sizeof(numNeighbors));
            neighbors.resize(numNeighbors);
            ifs.read((char*)neighbors.data(), numNeighbors * sizeof(neighbors[0]));
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