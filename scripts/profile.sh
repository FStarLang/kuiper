#!/bin/bash

set -euxo pipefail

# Usage validation
if [[ -z "$1" || ( "$1" != "0" && "$1" != "1" ) ]]; then
    echo "Usage: $0 <1 or 0>"
    echo "To enable NCU profiling: $0 1"
    echo "To disable NCU profiling: $0 0"
    exit 1
fi

NCU=$1

# Paths
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
PARENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE_DIR="$PARENT_DIR/profile_data"
NCU_REPORTS_DIR="$PROFILE_DIR/ncu_reports"
REPORTS_DIR="$PROFILE_DIR/reports"

# Ensure required directories exist
mkdir -p "$NCU_REPORTS_DIR" "$REPORTS_DIR"

# Parameters
LAPS=10
TILE=8

DIMS=("2048" "4096" "8192" "16384" "9216" "10240")

METRICS=(
    smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
    smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
    smsp__sass_inst_executed_op_shared_ld.sum
    smsp__sass_inst_executed_op_shared_st.sum
    smsp__sass_data_bytes_mem_local_op_ld.sum
    smsp__sass_inst_executed_op_global_ld.sum
    smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldgsts.sum
    smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
    l1tex__t_bytes_lookup_miss.sum
    sm__inst_issued.sum
    l1tex__t_sector_hit_rate
    lts__t_sector_hit_rate
    lts__t_request_hit_rate
    sm__inst_executed.sum
)

# Flatten metrics into comma-separated format
METRICS_STR=$(IFS=,; echo "${METRICS[*]}")

# Executables
EXECUTABLES=(
    "Test_Kuiper_GEMM_Naive__F32.exe"
    "Test_Kuiper_GEMM_Naive2__F32.exe"
    "Test_Kuiper_GEMM_Naive__U64.exe"
    "Test_Kuiper_GEMM_Naive2__U64.exe"
)

TILED_EXECUTABLES=(
    "Test_Kuiper_GEMM_BlockTiling1D__F32.exe"
    "Test_Kuiper_GEMM_BlockTiling1D__F64.exe"
    "Test_Kuiper_GEMM_BlockTiling1D__U32.exe"
    "Test_Kuiper_GEMM_BlockTiling1D__U64.exe"
    "Test_Kuiper_GEMM_SHMem__F32.exe"
    "Test_Kuiper_GEMM_SHMem__F64.exe"
    "Test_Kuiper_GEMM_SHMem__U32.exe"
    "Test_Kuiper_GEMM_SHMem__U64.exe"
    "Test_Kuiper_GEMM_Tiled__F32.exe"
    "Test_Kuiper_GEMM_Tiled__F64.exe"
    "Test_Kuiper_GEMM_Tiled__U32.exe"
    "Test_Kuiper_GEMM_Tiled__U64.exe"
)

# Check if ncu is available when profiling is enabled
if [[ "$NCU" -eq 1 && ! $(command -v ncu) ]]; then
    echo "Error: ncu not found. Ensure Nsight Compute CLI is installed."
    exit 1
fi

# Function to run profiling
run_profiling() {
    local exe=$1
    local dim=$2
    local ncu_enabled=$3

    base_name="${exe%.exe}"

    if [[ "$ncu_enabled" -eq 1 ]]; then
        echo "Profiling with NCU: $exe (DIM=$dim)"
        ncu --metrics "$METRICS_STR" "./obj/$exe" "$LAPS" "$dim" "$dim" "$dim" 0 &> "$NCU_REPORTS_DIR/${base_name}_${dim}_NCU_Profile"
    else
        echo "Running: $exe (DIM=$dim)"
        "./obj/$exe" "$LAPS" "$dim" "$dim" "$dim" 0 &> "$REPORTS_DIR/${base_name}_${dim}_Profile"
    fi
}

run_profiling_on_tiled() {
    local exe=$1
    local dim=$2
    local tile=$3
    local ncu_enabled=$4

    base_name="${exe%.exe}"

    if [[ "$ncu_enabled" -eq 1 ]]; then
        echo "Profiling with NCU: $exe (DIM=$dim, TILE=$tile)"
        ncu --metrics "$METRICS_STR" "./obj/$exe" "$LAPS" "$dim" "$dim" "$dim" "$tile" 0 &> "$NCU_REPORTS_DIR/${base_name}_${dim}_NCU_Profile"
    else
        echo "Running: $exe (DIM=$dim, TILE=$tile)"
        "./obj/$exe" "$LAPS" "$dim" "$dim" "$dim" "$tile" 0 &> "$REPORTS_DIR/${base_name}_${dim}_Profile"
    fi
}

# Main loop
for dim in "${DIMS[@]}"; do
    echo "========================= START $dim ========================="

    # Run profiling for standard executables
    for exe in "${EXECUTABLES[@]}"; do
        if [[ -x "./obj/$exe" ]]; then
            run_profiling "$exe" "$dim" "$NCU"
        else
            echo "Warning: Executable $exe not found or not executable."
        fi
    done

    # Run profiling for tiled executables
    for exe in "${TILED_EXECUTABLES[@]}"; do
        if [[ -x "./obj/$exe" ]]; then
            run_profiling_on_tiled "$exe" "$dim" "$TILE" "$NCU"
        else
            echo "Warning: Executable $exe not found or not executable."
        fi
    done

    echo "========================= END $dim ========================="
done

echo "Profiling completed successfully."
exit 0
