#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdint>

struct VectorData
{
    int id;
    std::vector<float> values;
};

class Dataset
{
public:
    std::vector<VectorData> vectors;
    int dimension = 0;
    int count = 0;

    void load(const std::string &path, const std::string &type);
};

// Helper: read 32-bit big-endian integer
inline uint32_t readBigEndian(std::ifstream &f)
{
    unsigned char bytes[4];
    f.read((char *)bytes, 4);
    return (uint32_t(bytes[0]) << 24) | (uint32_t(bytes[1]) << 16) |
           (uint32_t(bytes[2]) << 8) | (uint32_t(bytes[3]));
}

inline void Dataset::load(const std::string &path, const std::string &type)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open file: " + path);

    if (type == "mnist")
    {
        uint32_t magic = readBigEndian(f);
        if (magic == 2051)
        { // MNIST images
            uint32_t n = readBigEndian(f);
            uint32_t rows = readBigEndian(f);
            uint32_t cols = readBigEndian(f);
            dimension = rows * cols;
            count = n;
            vectors.resize(n);

            for (uint32_t i = 0; i < n; ++i)
            {
                vectors[i].id = i;
                vectors[i].values.resize(dimension);
                for (uint32_t j = 0; j < dimension; ++j)
                {
                    unsigned char val;
                    f.read((char *)&val, 1);
                    vectors[i].values[j] = static_cast<float>(val) / 255.0f;
                }
            }
        }
    }
    else if (type == "sift")
    {
        uint32_t n = readBigEndian(f);
        uint32_t dim = readBigEndian(f);
        dimension = dim;
        count = n;
        vectors.resize(n);

        for (uint32_t i = 0; i < n; ++i)
        {
            vectors[i].id = i;
            vectors[i].values.resize(dimension);
            for (uint32_t j = 0; j < dimension; ++j)
            {
                float val;
                f.read((char *)&val, sizeof(float));
                vectors[i].values[j] = val;
            }
        }
    }
    else
    {
        // // Plain text format: first line = n d
        // f.close();
        // f.open(path);
        // if (!f) throw std::runtime_error("Cannot open file: " + path);
        // int n = 0, d = 0;
        // f >> n >> d;
        // dimension = d;
        // count = n;
        // vectors.resize(n);
        // for (int i = 0; i < n; ++i) {
        //     vectors[i].id = i;
        //     vectors[i].values.resize(d);
        //     for (int j = 0; j < d; ++j) f >> vectors[i].values[j];
        // }
        throw std::runtime_error("Unsupported file format or type: " + type);
    }
}
