#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Config: edit these paths
# -----------------------------
BIN=./bin/search
DATA=./datasets/MNIST/train-images.idx3-ubyte
QUER=./datasets/MNIST/t10k-images.idx3-ubyte
OUTDIR=./runs
TYPE=mnist

mkdir -p "$OUTDIR"

# Build
echo "==> Building..."
make -j

# Helper: parse metrics from output file
# Prints: AF,Recall,QPS,tApprox,tTrue
parse_metrics () {
  local file="$1"
  local AF=$(grep -m1 "^Average AF:" "$file" | awk '{print $3}')
  local RC=$(grep -m1 "^Recall@N:" "$file" | awk '{print $2}')
  local QP=$(grep -m1 "^QPS:" "$file" | awk '{print $2}')
  local TA=$(grep -m1 "^tApproximateAverage:" "$file" | awk '{print $2}')
  local TT=$(grep -m1 "^tTrueAverage:" "$file" | awk '{print $2}')
  echo "${AF},${RC},${QP},${TA},${TT}"
}

# Helper: parse LSH diagnostics from STDERR capture
# Expects a line like:
# [LSH-DIAG] w=6 t_range=[0.001,5.997] min_h_seen=-123 neg_h_count=42
parse_lsh_diag () {
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

SUMMARY="${OUTDIR}/summary.csv"
echo "algo,params,AF,Recall,QPS,tApprox,tTrue,w,t_min,t_max,min_h_seen,neg_h_count" > "$SUMMARY"

run_lsh () {
  local seeds=("1" "2")
  local ks=("2" "3" "4")
  local Ls=("5" "10" "15")
  local ws=("2.0" "4.0" "6.0" "8.0")

  for seed in "${seeds[@]}"; do
    for k in "${ks[@]}"; do
      for L in "${Ls[@]}"; do
        for w in "${ws[@]}"; do
          local tag="lsh_s${seed}_k${k}_L${L}_w${w}"
          local out="${OUTDIR}/${tag}.txt"
          local err="${OUTDIR}/${tag}.err"
          echo "==> LSH ${tag}"
          # Enable diagnostics
          LSH_DEBUG=1 "$BIN" \
            -d "$DATA" -q "$QUER" -o "$out" -type "$TYPE" \
            -lsh -k "$k" -L "$L" -w "$w" -N 1 -R 2000 -seed "$seed" \
            2> "$err"

          local metrics
          metrics=$(parse_metrics "$out")
          local diag
          diag=$(parse_lsh_diag "$err")

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
}

run_cube () {
  local seeds=("1")
  local kprojs=("6" "8" "10")
  local Ms=("500" "1000")
  local probes=("1000" "3000" "5000")

  for seed in "${seeds[@]}"; do
    for kp in "${kprojs[@]}"; do
      for M in "${Ms[@]}"; do
        for pr in "${probes[@]}"; do
          local tag="cube_s${seed}_kproj${kp}_M${M}_probes${pr}"
          local out="${OUTDIR}/${tag}.txt"
          echo "==> Hypercube ${tag}"
          "$BIN" \
            -d "$DATA" -q "$QUER" -o "$out" -type "$TYPE" \
            -hypercube -kproj "$kp" -M "$M" -probes "$pr" -N 1 -R 2000 -seed "$seed"
          local metrics
          metrics=$(parse_metrics "$out")
          echo "Hypercube,kproj=${kp};M=${M};probes=${pr};seed=${seed},${metrics},,,,,," >> "$SUMMARY"
        done
      done
    done
  done
}

run_ivfflat () {
  local seeds=("1")
  local kcs=("32" "64" "128")
  local nprobes=("4" "8" "16")

  for seed in "${seeds[@]}"; do
    for kc in "${kcs[@]}"; do
      for np in "${nprobes[@]}"; do
        local tag="ivfflat_s${seed}_kc${kc}_np${np}"
        local out="${OUTDIR}/${tag}.txt"
        echo "==> IVFFlat ${tag}"
        "$BIN" \
          -d "$DATA" -q "$QUER" -o "$out" -type "$TYPE" \
          -ivfflat -kclusters "$kc" -nprobe "$np" -N 1 -R 2000 -seed "$seed"
        local metrics
        metrics=$(parse_metrics "$out")
        echo "IVFFlat,kclusters=${kc};nprobe=${np};seed=${seed},${metrics},,,,,," >> "$SUMMARY"
      done
    done
  done
}

run_ivfpq () {
  local seeds=("1")
  local kcs=("50" "64")
  local nprobes=("4" "8")
  local Msubs=("8" "16")
  local nbits=("8")

  for seed in "${seeds[@]}"; do
    for kc in "${kcs[@]}"; do
      for np in "${nprobes[@]}"; do
        for Msub in "${Msubs[@]}"; do
          for nb in "${nbits[@]}"; do
            local tag="ivfpq_s${seed}_kc${kc}_np${np}_M${Msub}_nb${nb}"
            local out="${OUTDIR}/${tag}.txt"
            echo "==> IVFPQ ${tag}"
            "$BIN" \
              -d "$DATA" -q "$QUER" -o "$out" -type "$TYPE" \
              -ivfpq -kclusters "$kc" -nprobe "$np" -Msub "$Msub" -nbits "$nb" \
              -N 1 -R 2000 -seed "$seed"
            local metrics
            metrics=$(parse_metrics "$out")
            echo "IVFPQ,kclusters=${kc};nprobe=${np};M=${Msub};nbits=${nb};seed=${seed},${metrics},,,,,," >> "$SUMMARY"
          done
        done
      done
    done
  done
}

# Run all batches
run_lsh
run_cube
run_ivfflat
run_ivfpq

echo "==> Done. Summary: $SUMMARY"
