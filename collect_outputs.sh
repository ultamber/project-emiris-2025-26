#!/usr/bin/env bash
set -euo pipefail

# Directory containing output files
OUTDIR="./runs"

# Check directory exists
if [[ ! -d "$OUTDIR" ]]; then
  echo "Error: directory '$OUTDIR' not found." >&2
  exit 1
fi

# Collect all .txt outputs into an array
echo "==> Collecting outputs from: $OUTDIR"

# Declare an array
declare -a outputs=()

# Iterate over all text files
while IFS= read -r -d '' file; do
  outputs+=("$file")
done < <(find "$OUTDIR" -type f -name "*.txt" -print0 | sort -z)

echo "Found ${#outputs[@]} output files."

# Optional: preview file names
for f in "${outputs[@]}"; do
  echo "  -> $f"
done

# Example: extract metrics (e.g. Average AF and Recall) from each file
echo
echo "==> Extracting metrics into Bash arrays..."

declare -a avgAF recall qps
for f in "${outputs[@]}"; do
  AF=$(grep -m1 "^Average AF:" "$f" | awk '{print $3}')
  RC=$(grep -m1 "^Recall@N:" "$f" | awk '{print $2}')
  QP=$(grep -m1 "^QPS:" "$f" | awk '{print $2}')
  avgAF+=("$AF")
  recall+=("$RC")
  qps+=("$QP")
done

# Print summary arrays
echo "Average AF values: ${avgAF[*]}"
echo "Recall values:     ${recall[*]}"
echo "QPS values:        ${qps[*]}"

# Example: compute average AF across all runs
if [[ ${#avgAF[@]} -gt 0 ]]; then
  avg_total=$(awk -v arr="${avgAF[*]}" 'BEGIN {
    split(arr, a, " "); s=0; for (i in a) s+=a[i]; print s/length(a)
  }')
  echo "Overall average AF: $avg_total"
fi
