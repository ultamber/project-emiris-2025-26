#ifndef BRUTEFORCE_HPP
#define BRUTEFORCE_HPP

#include <fstream>
#include "dataset.hpp"
#include "arguments.hpp"

class BruteForce {
public:
    explicit BruteForce(const Arguments& args);
    void buildIndex(const Dataset& data); // kept for API symmetry (no-op)
    void search(const Dataset& queries, std::ofstream& out);

private:
    Arguments cfg;
    // reference dataset kept for search (copy or pointer to original)
    const Dataset* indexedData = nullptr;

    // helpers
    static double euclidean_l2(const std::vector<float>& a, const std::vector<float>& b);
};

#endif // BRUTEFORCE_HPP
