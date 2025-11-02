#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <numeric>
#include <cmath>

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
                for (int j = 0; j < dimension; ++j)
                {
                    unsigned char val;
                    f.read((char *)&val, 1);
                    vectors[i].values[j] = static_cast<float>(val) / 255.0f;
                }

                // Normalize the entire image vector to unit L2 norm
                double norm = std::sqrt(std::inner_product(
                    vectors[i].values.begin(),
                    vectors[i].values.end(),
                    vectors[i].values.begin(),
                    0.0));

                if (norm > 1e-12)
                { // prevent division by zero
                    for (auto &x : vectors[i].values)
                        x /= static_cast<float>(norm);
                }
            }
        }
    }
    else if (type == "sift")
    {
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        f.seekg(0, std::ios::beg);

        int dim;
        f.read((char *)&dim, 4);

        if (dim <= 0 || dim > 10000)
            throw std::runtime_error("Invalid dimension read from fvecs file");

        size_t vector_size = 4 + sizeof(float) * dim;
        size_t n = file_size / vector_size;

        dimension = dim;
        count = static_cast<int>(n);
        vectors.resize(count);

        f.seekg(0, std::ios::beg);

        for (size_t i = 0; i < n; ++i)
        {
            int d;
            f.read((char *)&d, 4);
            if (d != dim)
                throw std::runtime_error("Inconsistent vector dimension in .fvecs file");

            vectors[i].id = static_cast<int>(i);
            vectors[i].values.resize(dim);
            f.read((char *)vectors[i].values.data(), dim * sizeof(float));

            // Optional: rescale SIFT descriptors to make Euclidean LSH bins meaningful
            for (auto &x : vectors[i].values)
                x *= 100.0f; // scale factor; adjust if needed
        }
    }
    else
    {
        throw std::runtime_error("Unsupported file format or type: " + type);
    }
}
