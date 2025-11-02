#include <iostream>
#include <fstream>
#include <iomanip>
#include "arguments.hpp"
#include "dataset.hpp"
#include "search_method.hpp"
#include "lsh.hpp"
#include "hypercube.hpp"
#include "ivfflat.hpp"
#include "ivfpq.hpp"
Arguments parseArgs(int argc, char *argv[])
{
    Arguments a;
    for (int i = 1; i < argc; ++i)
    {
        std::string flag = argv[i];
        if (flag == "-d" && i + 1 < argc)
            a.inputFile = argv[++i];
        else if (flag == "-q" && i + 1 < argc)
            a.queryFile = argv[++i];
        else if (flag == "-o" && i + 1 < argc)
            a.outputFile = argv[++i];
        else if (flag == "-type" && i + 1 < argc)
            a.type = argv[++i];
        else if (flag == "-lsh")
            a.useLSH = true;
        else if (flag == "-hypercube")
            a.useHypercube = true;
        else if (flag == "-ivfflat")
            a.useIVFFlat = true;
        else if (flag == "-ivfpq")
            a.useIVFPQ = true;
        else if (flag == "-bruteforce")
            a.useBruteForce = true;
        else if (flag == "-N" && i + 1 < argc)
            a.N = std::stoi(argv[++i]);
        else if (flag == "-R" && i + 1 < argc)
            a.R = std::stod(argv[++i]);
        else if (flag == "-seed" && i + 1 < argc)
            a.seed = std::stoi(argv[++i]);
        else if (flag == "-range" && i + 1 < argc)
        {
            std::string v = argv[++i];
            a.rangeSearch = (v == "true" || v == "1");
        }
        else if (flag == "-k" && i + 1 < argc)
            a.k = std::stoi(argv[++i]);
        else if (flag == "-L" && i + 1 < argc)
            a.L = std::stoi(argv[++i]);
        else if (flag == "-w" && i + 1 < argc)
            a.w = std::stod(argv[++i]);
        else if (flag == "-kproj" && i + 1 < argc)
            a.kproj = std::stoi(argv[++i]);
        else if (flag == "-M" && i + 1 < argc)
            a.M = std::stoi(argv[++i]);
        else if (flag == "-probes" && i + 1 < argc)
            a.probes = std::stoi(argv[++i]);
        else if (flag == "-kclusters" && i + 1 < argc)
            a.kclusters = std::stoi(argv[++i]);
        else if (flag == "-nprobe" && i + 1 < argc)
            a.nprobe = std::stoi(argv[++i]);
        else if (flag == "-nbits" && i + 1 < argc)
            a.nbits = std::stoi(argv[++i]);
        else if (flag == "-Msub" && i + 1 < argc)
            a.Msubvectors = std::stoi(argv[++i]);
    }
    return a;
}

int main(int argc, char *argv[])
{
    if (argc < 5)
    {
        std::cerr << "Usage: ./search -d <input> -q <query> -o <output> -type mnist -lsh|-hypercube|-ivfflat|-ivfpq [params]\n";
        return 1;
    }

    Arguments args = parseArgs(argc, argv);
    std::cout << "Input: " << args.inputFile << "\nQuery: " << args.queryFile
              << "\nOutput: " << args.outputFile << "\n";

    Dataset data, queries;
    data.load(args.inputFile, args.type);
    queries.load(args.queryFile, args.type);

    std::ofstream out(args.outputFile);
    if (!out)
    {
        std::cerr << "Cannot open output file.\n";
        return 1;
    }

    if (args.useLSH)
    {
        LSH alg(args);
        alg.buildIndex(data);
        alg.search(queries, out);
    }
    else if (args.useHypercube)
    {
        Hypercube alg(args);
        alg.buildIndex(data);
        alg.search(queries, out);
    }
    else if (args.useIVFFlat)
    {
        IVFFlat alg(args);
        alg.buildIndex(data);
        // double sil = alg.silhouetteScore();
        // std::cout << std::fixed << std::setprecision(6);
        // std::cout << "\n=== Silhouette Evaluation ===\n";
        // std::cout << "Overall silhouette coefficient: " << sil << "\n";

        // auto perCluster = alg.silhouettePerCluster();
        // for (size_t i = 0; i < perCluster.size(); ++i)
        //     std::cout << "Cluster " << i << ": silhouette = " << perCluster[i] << "\n";
        // std::cout << "=============================\n";
        alg.search(queries, out);
    }
    else if (args.useIVFPQ)
    {
        IVFPQ alg(args);
        alg.buildIndex(data);
        // double sil = alg.silhouetteScore();
        // std::cout << std::fixed << std::setprecision(6);
        // std::cout << "\n=== Silhouette Evaluation ===\n";
        // std::cout << "Overall silhouette coefficient: " << sil << "\n";

        // auto perCluster = alg.silhouettePerCluster();
        // for (size_t i = 0; i < perCluster.size(); ++i)
        //     std::cout << "Cluster " << i << ": silhouette = " << perCluster[i] << "\n";
        // std::cout << "=============================\n";
        alg.search(queries, out);
    }
    else
    {
        std::cerr << "Error: specify -lsh or -hypercube or -ivfflat or -ivfpq\n";
        return 1;
    }

    std::cout << "Search completed.\n";
    return 0;
}
