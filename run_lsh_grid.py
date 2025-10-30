#!/usr/bin/env python3
import subprocess
import csv
import time
import os

# === CONFIGURATION ===
BIN_PATH = "./bin/search"
DATASET = "./datasets/SIFT/sift_base.fvecs"
QUERIES = "./datasets/SIFT/sift_query.fvecs"
OUTPUT_DIR = "./outputs/"
TYPE = "sift"

# LSH parameter grid
K_VALUES = [2, 3, 4]
L_VALUES = [5, 10, 15]
W_VALUES = [2.0, 4.0, 6.0, 8.0]
SEEDS = [1, 2]

# number of nearest neighbors
N = 1
R = 200.0

SUMMARY_FILE = "./outputs/summary_lsh_grid.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_metrics(output_text):
    """Extract metrics from program stdout."""
    metrics = {
        "AF": None,
        "Recall": None,
        "QPS": None,
        "tApprox": None,
        "tTrue": None,
    }
    for line in output_text.splitlines():
        if "Average AF" in line:
            metrics["AF"] = float(line.split(":")[1].strip())
        elif "Recall@N" in line:
            metrics["Recall"] = float(line.split(":")[1].strip())
        elif "QPS" in line:
            metrics["QPS"] = float(line.split(":")[1].strip())
        elif "tApproximateAverage" in line:
            metrics["tApprox"] = float(line.split(":")[1].strip())
        elif "tTrueAverage" in line:
            metrics["tTrue"] = float(line.split(":")[1].strip())
    return metrics


def run_search(k, L, w, seed):
    """Run one experiment."""
    outfile = os.path.join(OUTPUT_DIR, f"lsh_k{k}_L{L}_w{w}_s{seed}.txt")
    cmd = [
        BIN_PATH,
        "-d", DATASET,
        "-q", QUERIES,
        "-o", outfile,
        "-type", TYPE,
        "-lsh",
        "-k", str(k),
        "-L", str(L),
        "-w", str(w),
        "-N", str(N),
        "-R", str(R),
        "-seed", str(seed),
    ]
    print(f"\n>>> Running LSH with k={k}, L={L}, w={w}, seed={seed}")
    t0 = time.time()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    dt = time.time() - t0
    metrics = parse_metrics(result.stdout)
    metrics["runtime"] = dt
    return metrics, result.stdout


def main():
    # CSV header
    with open(SUMMARY_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algo", "params", "AF", "Recall", "QPS", "tApprox", "tTrue", "runtime"])

    # parameter grid
    for k in K_VALUES:
        for L in L_VALUES:
            for w in W_VALUES:
                for seed in SEEDS:
                    metrics, stdout = run_search(k, L, w, seed)
                    params = f"k={k};L={L};w={w};seed={seed}"
                    row = [
                        "LSH",
                        params,
                        metrics["AF"],
                        metrics["Recall"],
                        metrics["QPS"],
                        metrics["tApprox"],
                        metrics["tTrue"],
                        metrics["runtime"],
                    ]
                    with open(SUMMARY_FILE, "a", newline="") as f:
                        csv.writer(f).writerow(row)

    print("\n✅ All experiments complete. Results saved to:", SUMMARY_FILE)


if __name__ == "__main__":
    main()
