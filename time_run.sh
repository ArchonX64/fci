#!/usr/bin/env bash
# Time how long a Python script takes to run.
#
# Usage: ./time_run.sh [-n RUNS] [-p RUNNER] script.py [script args...]
#   -n RUNS    number of times to run the script (default: 1)
#   -p RUNNER  command used to run the script (default: "uv run")

set -euo pipefail

runs=1
runner=(uv run)

while getopts ":n:p:h" opt; do
    case $opt in
        n) runs=$OPTARG ;;
        p) read -ra runner <<< "$OPTARG" ;;
        h) sed -n '2,7p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: -$OPTARG" >&2; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 [-n RUNS] [-p RUNNER] script.py [script args...]" >&2
    exit 1
fi

script=$1
shift

if [[ ! -f $script ]]; then
    echo "Error: '$script' not found" >&2
    exit 1
fi

# Sync the uv environment once up front so it isn't counted in the timings
if [[ ${runner[0]} == uv ]]; then
    uv sync --quiet
fi

times=()
for ((i = 1; i <= runs; i++)); do
    start=$(date +%s.%N)
    "${runner[@]}" "$script" "$@"
    end=$(date +%s.%N)
    elapsed=$(awk -v s="$start" -v e="$end" 'BEGIN { printf "%.3f", e - s }')
    times+=("$elapsed")
    echo "Run $i: ${elapsed}s" >&2
done

if ((runs > 1)); then
    printf '%s\n' "${times[@]}" | awk '
        NR == 1 { min = max = $1 }
        { sum += $1; if ($1 < min) min = $1; if ($1 > max) max = $1 }
        END { printf "Runs: %d  mean: %.3fs  min: %.3fs  max: %.3fs\n", NR, sum / NR, min, max }
    ' >&2
fi
