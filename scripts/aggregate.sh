#!/bin/bash

set -euo pipefail

# Paths
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
PARENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE_DIR="$PARENT_DIR/profile_data"
NCU_REPORTS_DIR="$PROFILE_DIR/ncu_reports"
REPORTS_DIR="$PROFILE_DIR/reports"

# Validate directories
if [[ ! -d "$NCU_REPORTS_DIR" && ! -d "$REPORTS_DIR" ]]; then
    echo "Error: Neither $NCU_REPORTS_DIR nor $REPORTS_DIR found."
    exit 1
fi

# Output CSV file
TIMESTAMP=$(date '+%Y-%m-%d_%H:%M:%S')
OUTPUT_FILE="${PROFILE_DIR}/summary_all_files_${TIMESTAMP}.csv"
echo "Filename,Avg Wall Time (s),Median Wall Time (s),Std Dev Wall Time (s),Avg CPU Time (s),Median CPU Time (s),Std Dev CPU Time (s),Avg GFLOPS,Median GFLOPS,Std Dev GFLOPS" > "$OUTPUT_FILE"

# Identify all files
FILES=()
for dir in "$REPORTS_DIR"/*; do
    [[ -d "$dir" ]] || continue
    for file in "$dir"/*; do
        [[ -f "$file" ]] && FILES+=("$file")
    done
done

TOTAL_FILES=${#FILES[@]}
if [[ ${TOTAL_FILES} -eq 0 ]]; then
    echo "No files found."
    exit 1
fi

printf "Found %d files.\n" "${TOTAL_FILES}"

# Loop through each file with progress counter
counter=0
for file in "${FILES[@]}"; do
    [[ -f "$file" ]] || continue

    counter=$((counter + 1))
    # Extract filename
    filename=$(basename "$file")

    # Show progress
    echo "[$counter/$TOTAL_FILES] Processing $filename..."

    # Extract values
    wall_times=()
    cpu_times=()
    gflops=()

    while IFS= read -r line; do
        if [[ "$line" =~ "wall time" ]]; then
            wall_time=$(echo "$line" | grep -oP '(?<=wall time = )[0-9.]+')
            [[ -n "$wall_time" ]] && wall_times+=("$wall_time")
        fi

        if [[ "$line" =~ "CPU time" ]]; then
            cpu_time=$(echo "$line" | grep -oP '(?<=in )[0-9.]+(?=s total CPU time)')
            [[ -n "$cpu_time" ]] && cpu_times+=("$cpu_time")
        fi

        if [[ "$line" =~ "Estimated GFLOPS" ]]; then
            gflops_value=$(echo "$line" | grep -oP '(?<=Estimated GFLOPS: )[0-9.]+')
            [[ -n "$gflops_value" ]] && gflops+=("$gflops_value")
        fi
    done < "$file"

    # Use Python for calculations only if we have valid values
    if [[ ${#wall_times[@]} -gt 0 && ${#cpu_times[@]} -gt 0 && ${#gflops[@]} -gt 0 ]]; then
        # Pass Bash arrays as space-separated strings
        result=$(python3 ./scripts/calc_stats.py "${wall_times[*]}" "${cpu_times[*]}" "${gflops[*]}")

        # Append results to CSV
        echo "$filename,$result" >> "$OUTPUT_FILE"
    else
        echo "Skipping $file due to missing values."
    fi
done

# Sort CSV by filename for better readability
sort -t, -k1,1 "$OUTPUT_FILE" -o "$OUTPUT_FILE"

echo "Summary saved to $OUTPUT_FILE"
