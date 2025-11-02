#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config
# -----------------------------
BIN=./bin/search
TYPE=mnist
DATA=""
QUER=""
OUTDIR=./runs
mkdir -p "$OUTDIR"

# -----------------------------
# Parse flags
# -----------------------------
usage() {
  cat <<EOF
Usage: $0 [-t TYPE] [-d INPUT_FILE] [-q QUERY_FILE] [--job-id N --num-jobs M]

Flags:
  -t, --type TYPE       Dataset type: 'mnist' (default) or 'sift'
  -d, --data PATH       Path to input dataset file
  -q, --query PATH      Path to query dataset file
  --job-id N            Parallel job ID (1..M)
  --num-jobs M          Total number of parallel jobs
  -h, --help            Show this help message
EOF
  exit 0
}

JOB_ID=1
NUM_JOBS=1

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -t|--type) TYPE="$2"; shift 2 ;;
    -d|--data) DATA="$2"; shift 2 ;;
    -q|--query) QUER="$2"; shift 2 ;;
    --job-id) JOB_ID="$2"; shift 2 ;;
    --num-jobs) NUM_JOBS="$2"; shift 2 ;;
    -h|--help) usage ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; usage ;;
    *) echo "Positional args not supported."; usage ;;
  esac
done

# -----------------------------
# Validate defaults
# -----------------------------
if [[ "$TYPE" != "sift" && "$TYPE" != "mnist" ]]; then
  echo "Error: invalid type '$TYPE'. Use 'sift' or 'mnist'." >&2
  exit 1
fi

if [[ -z "${DATA:-}" ]]; then
  DATA="./datasets/${TYPE^^}/input.dat"
  [[ "$TYPE" == "mnist" ]] && DATA="./datasets/MNIST/input.dat"
fi
if [[ -z "${QUER:-}" ]]; then
  QUER="./datasets/${TYPE^^}/query.dat"
  [[ "$TYPE" == "mnist" ]] && QUER="./datasets/MNIST/query.dat"
fi

echo "Using dataset type: $TYPE"
echo "Input:  $DATA"
echo "Query:  $QUER"
echo "Job:    $JOB_ID / $NUM_JOBS"
echo

# -----------------------------
# Build binary
# -----------------------------

# -----------------------------
# Helpers
# -----------------------------
parse_metrics() {
  local file="$1"
  local AF=$(grep "^Average AF:" "$file" | tail -n1 | awk '{print $3}')
  local RC=$(grep "^Recall@N:" "$file" | tail -n1 | awk '{print $2}')
  local QP=$(grep "^QPS:" "$file" | tail -n1 | awk '{print $2}')
  local TA=$(grep "^tApproximateAverage:" "$file" | tail -n1 | awk '{print $2}')
  local TT=$(grep "^tTrueAverage:" "$file" | tail -n1 | awk '{print $2}')
  echo "${AF},${RC},${QP},${TA},${TT}"
}

parse_lsh_diag() {
  local errfile="$1"
  local line
  line=$(grep -m1 "^\[LSH-DIAG\]" "$errfile" || true)
  if [[ -z "$line" ]]; then
    echo ",,,,"
    return
  fi
  local W=$(echo "$line" | sed -E 's/.*w=([0-9.]+).*/\1/')
  local TRANGE=$(echo "$line" | sed -E 's/.*t_range=\[([0-9eE\.\+\-]+),([0-9eE\.\+\-]+)\].*/\1,\2/')
  local MINH=$(echo "$line" | sed -E 's/.*min_h_seen=([-0-9]+).*/\1/')
  local NEGC=$(echo "$line" | sed -E 's/.*neg_h_count=([0-9]+).*/\1/')
  echo "${W},${TRANGE},${MINH},${NEGC}"
}

SUMMARY="${OUTDIR}/summary_${TYPE}_job${JOB_ID}.csv"
echo "algo,params,AF,Recall,QPS,tApprox,tTrue,w,t_min,t_max,min_h_seen,neg_h_count" > "$SUMMARY"

# -----------------------------
# Work splitting helper
# -----------------------------
should_run() {
  ((job_index++))
  local mod=$(( (job_index - 1) % NUM_JOBS + 1 ))
  [[ $mod -eq $JOB_ID ]]
}

# -----------------------------
# Experiments
# -----------------------------
run_lsh() {
  local seeds=("1" "2")
  local ks=("2" "3" "4")
  local Ls=("5" "10" "15")
  local ws=("2.0" "4.0" "6.0" "8.0")
  job_index=0

  for seed in "${seeds[@]}"; do
    for k in "${ks[@]}"; do
      for L in "${Ls[@]}"; do
        for w in "${ws[@]}"; do
          should_run || continue
          local tag="lsh_s${seed}_k${k}_L${L}_w${w}"
          local out="${OUTDIR}/${tag}.txt"
          local err="${OUTDIR}/${tag}.err"
          echo "==> LSH ${tag}"
          LSH_DEBUG=1 "$BIN" -d "$DATA" -q "$QUER" -o "$out" -type "$TYPE" \
            -lsh -k "$k" -L "$L" -w "$w" -N 1 -R 2000 -seed "$seed" 2> "$err"
          local metrics=$(parse_metrics "$out")
          local diag=$(parse_lsh_diag "$err")
          echo "LSH,k=${k};L=${L};w=${w};seed=${seed},${metrics},${diag}" >> "$SUMMARY"
        done
      done
    done
  done
  echo
}

run_cube() {
  local seeds=("1")
  local kprojs=("6" "8" "10")
  local Ms=("500" "1000")
  local probes=("1000" "3000" "5000")
  job_index=0

  for seed in "${seeds[@]}"; do
    for kp in "${kprojs[@]}"; do
      for M in "${Ms[@]}"; do
        for pr in "${probes[@]}"; do
          should_run || continue
          local tag="cube_s${seed}_kproj${kp}_M${M}_probes${pr}"
          local out="${OUTDIR}/${tag}.txt"
          echo "==> Hypercube ${tag}"
          "$BIN" -d "$DATA" -q "$QUER" -o "$out" -type "$TYPE" \
            -hypercube -kproj "$kp" -M "$M" -probes "$pr" -N 1 -R 2000 -seed "$seed"
          local metrics=$(parse_metrics "$out")
          echo "Hypercube,kproj=${kp};M=${M};probes=${pr};seed=${seed},${metrics},,,,,," >> "$SUMMARY"
        done
      done
    done
  done
  echo
}

run_ivfflat() {
  local seeds=("1")
  local kcs=("32" "64" "128")
  local nprobes=("4" "8" "16")
  job_index=0

  for seed in "${seeds[@]}"; do
    for kc in "${kcs[@]}"; do
      for np in "${nprobes[@]}"; do
        should_run || continue
        local tag="ivfflat_s${seed}_kc${kc}_np${np}"
        local out="${OUTDIR}/${tag}.txt"
        echo "==> IVFFlat ${tag}"
        "$BIN" -d "$DATA" -q "$QUER" -o "$out" -type "$TYPE" \
          -ivfflat -kclusters "$kc" -nprobe "$np" -N 1 -R 2000 -seed "$seed"
        local metrics=$(parse_metrics "$out")
        echo "IVFFlat,kclusters=${kc};nprobe=${np};seed=${seed},${metrics},,,,,," >> "$SUMMARY"
      done
    done
  done
  echo
}

run_ivfpq() {
  local seeds=("1")
  local kcs=("50" "64")
  local nprobes=("4" "8")
  local Msubs=("8" "16")
  local nbits=("8")
  job_index=0

  for seed in "${seeds[@]}"; do
    for kc in "${kcs[@]}"; do
      for np in "${nprobes[@]}"; do
        for Msub in "${Msubs[@]}"; do
          for nb in "${nbits[@]}"; do
            should_run || continue
            local tag="ivfpq_s${seed}_kc${kc}_np${np}_M${Msub}_nb${nb}"
            local out="${OUTDIR}/${tag}.txt"
            echo "==> IVFPQ ${tag}"
            "$BIN" -d "$DATA" -q "$QUER" -o "$out" -type "$TYPE" \
              -ivfpq -kclusters "$kc" -nprobe "$np" -Msub "$Msub" -nbits "$nb" \
              -N 1 -R 2000 -seed "$seed"
            local metrics=$(parse_metrics "$out")
            echo "IVFPQ,kclusters=${kc};nprobe=${np};M=${Msub};nbits=${nb};seed=${seed},${metrics},,,,,," >> "$SUMMARY"
          done
        done
      done
    done
  done
  echo
}
echo "==> Computing ground truth (once)..."
GT_CACHE="groundtruth_${TYPE}_N1.bin"

if [ ! -f "$GT_CACHE" ]; then
  echo "Ground truth not cached, will be computed on first run"
else
  echo "Using cached ground truth: $GT_CACHE"
fi
# -----------------------------
# Run all
# -----------------------------
run_lsh
run_cube
run_ivfflat
run_ivfpq

echo "==> Done. Summary: $SUMMARY"