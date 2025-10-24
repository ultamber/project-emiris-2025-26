#include "dataset.hpp"
#include <cmath>
#include <limits>
#include <vector>
#include <chrono>
#include <algorithm>

inline double euclidean(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    return std::sqrt(sum);
}

// Brute-force search for the true nearest neighbor
inline std::pair<int,double> trueNearest(const std::vector<float>& q, const Dataset& data) {
    double best = std::numeric_limits<double>::infinity();
    int bestId = -1;
    for (const auto& v : data.vectors) {
        double d = euclidean(q, v.values);
        if (d < best) {
            best = d;
            bestId = v.id;
        }
    }
    return {bestId, best};
}
