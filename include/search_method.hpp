#pragma once
#include "arguments.hpp"
#include "dataset.hpp"
#include <fstream>

class SearchMethod {
public:
    explicit SearchMethod(const Arguments& a) : args(a) {}
    virtual ~SearchMethod() = default;

    virtual void buildIndex(const Dataset& data) = 0;
    virtual void search(const Dataset& queries, std::ofstream& out) = 0;

protected:
    Arguments args;
};
