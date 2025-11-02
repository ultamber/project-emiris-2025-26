#pragma once
#include "arguments.hpp"
#include "dataset.hpp"
#include <fstream>
#include "ground_truth.hpp"
class SearchMethod {
public:
    explicit SearchMethod(const Arguments& a) : args(a) {}
    virtual ~SearchMethod() = default;

    virtual void buildIndex(const Dataset& data) = 0;
    virtual void search(const Dataset& queries, std::ofstream& out ,const GroundTruth *groundTruth = nullptr) = 0;

protected:
    Arguments args;
};
