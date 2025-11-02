#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config: edit these paths
# -----------------------------
BIN=./bin/search

# Defaults (can be overridden via flags)
TYPE="mnist"
# leave DATA/QUER/R_PARAM/OUTDIR empty so we set them after parsing TYPE (and after validating lowercase)
DATA=""
QUER=""
R_PARAM=""
OUTDIR=""

# Parse optional parameters (flags only):
# Usage: run_grid.sh -t TYPE -d DATA -q QUERY
usage() {
  cat <<EOF
Usage: $0 [-t TYPE] [-d INPUT_FILE] [-q QUERY_FILE]

Flags (all optional):
  -t, --type TYPE       Dataset type: 'mnist' (default) or 'sift' (only lowercase accepted)
  -d, --data PATH       Path to input dataset file (overrides default for the chosen type)
  -q, --query PATH      Path to query dataset file (overrides default for the chosen type)
  --job-id N            Parallel job ID (1..M)
  --num-jobs M          Total number of parallel jobs
  -h, --help            Show this help message
EOF
  exit 0
}

JOB_ID=1
NUM_JOBS=1

# Quick --help
for a in "$@"; do
  case "$a" in
    -h|--help) usage ;;
  esac
done

# Parse flags
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -t|--type) TYPE="$2"; shift 2 ;;
    -d|--data) DATA="$2"; shift 2 ;;
    -q|--query) QUER="$2"; shift 2 ;;
    --job-id) JOB_ID="$2"; shift 2 ;;
    --num-jobs) NUM_JOBS="$2"; shift 2 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; usage ;;
    *) echo "Positional arguments are not supported."; usage ;;
  esac
done

# Enforce lowercase-only TYPE values and valid choices
if [[ "$TYPE" != "sift" && "$TYPE" != "mnist" ]]; then
  echo "Error: Invalid type '$TYPE'. Supported lowercase types: 'sift', 'mnist'." >&2
  exit 1
fi

# If DATA/QUER/OUTDIR were not explicitly provided, set sensible defaults based on TYPE
if [[ "$TYPE" == "sift" ]]; then
  R_PARAM=2
else
  R_PARAM=2000
fi

if [[ -z "${OUTDIR:-}" ]]; then
  OUTDIR="./runs/${TYPE^^}"
  mkdir -p "$OUTDIR"
fi
if [[ -z "${DATA:-}" ]]; then
  DATA="./datasets/${TYPE^^}/input.dat"
fi
if [[ -z "${QUER:-}" ]]; then
    QUER="./datasets/${TYPE^^}/query.dat"
fi

echo "Using dataset type: $TYPE"
echo "Input:  $DATA"
echo "Query:  $QUER"
echo "Job:    $JOB_ID / $NUM_JOBS"
echo

# Build
echo "==> Building..."
make -j
echo

# Helper: parse metrics from output file
# Prints: AF,Recall,QPS,tApprox,tTrue
parse_metrics() {
  local file="$1"
  local AF=$(grep "^Average AF:" "$file" | tail -n1 | awk '{print $3}')
  local RC=$(grep "^Recall@N:" "$file" | tail -n1 | awk '{print $2}')
  local QP=$(grep "^QPS:" "$file" | tail -n1 | awk '{print $2}')
  local TA=$(grep "^tApproximateAverage:" "$file" | tail -n1 | awk '{print $2}')
  local TT=$(grep "^tTrueAverage:" "$file" | tail -n1 | awk '{print $2}')
  echo "${AF},${RC},${QP},${TA},${TT}"
}

# Helper: parse LSH diagnostics from STDERR capture
# Expects a line like:
# [LSH-DIAG] w=6 t_range=[0.001,5.997] min_h_seen=-123 neg_h_count=42
parse_lsh_diag() {
  local errfile="$1"
  local line
  line=$(grep -m1 "^\[LSH-DIAG\]" "$errfile" || true)
  if [[ -z "$line" ]]; then
    echo ",,,," # empty columns if not found
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
          "$BIN" -d "$DATA" -q "$QUER" -o "$out" -type "$TYPE" \
            -lsh -k "$k" -L "$L" -w "$w" -N 1 -R "$R_PARAM" -seed "$seed" 2> "$err"
          local metrics=$(parse_metrics "$out")
          local diag=$(parse_lsh_diag "$err")

          # Check t-range conforms to [0,w)
          # Warn if outside (shouldn't happen)
          IFS=',' read -r _w _tmin _tmax _minh _negc <<< "$diag"
          if [[ -n "$_tmin" && -n "$_tmax" ]]; then
            awk -v tmin="$_tmin" -v tmax="$_tmax" -v w="$w" '
              BEGIN {
                if (tmin < 0 - 1e-9 || tmax > w + 1e-9) {
                  printf("WARNING: t outside [0,w): tmin=%.6f tmax=%.6f w=%.6f\n", tmin, tmax, w) > "/dev/stderr";
                }
              }' </dev/null
          fi

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
            -hypercube -kproj "$kp" -M "$M" -probes "$pr" -N 1 -R "$R_PARAM" -seed "$seed"
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
          -ivfflat -kclusters "$kc" -nprobe "$np" -N 1 -R "$R_PARAM" -seed "$seed"
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
              -N 1 -R "$R_PARAM" -seed "$seed"
            local metrics=$(parse_metrics "$out")
            echo "IVFPQ,kclusters=${kc};nprobe=${np};M=${Msub};nbits=${nb};seed=${seed},${metrics},,,,,," >> "$SUMMARY"
          done
        done
      done
    done
  done
  echo
}

# -----------------------------
# Run all
# -----------------------------
run_lsh
run_cube
run_ivfflat
run_ivfpq

echo "==> Done. Summary: $SUMMARY"