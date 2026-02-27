#!/bin/bash
#
# run_tests.sh — Build, run, and report all MPCC test programs.
#
# Generates timestamped output directories with CSV logs and PNG plots.
# Uses acados MPCC solver.
#
# Usage:
#   cd /home/stephen/quanser-acc/cpp/test_build
#   bash run_tests.sh              # Run all tests
#   bash run_tests.sh --quick      # Only test_mpcc_solver (fast)
#   bash run_tests.sh --verbose    # Print full test output

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$SCRIPT_DIR/results/$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

QUICK=false
VERBOSE=false
DIAGNOSTICS=false
REALISTIC=false
for arg in "$@"; do
    case "$arg" in
        --quick)       QUICK=true ;;
        --verbose)     VERBOSE=true ;;
        --diagnostics) DIAGNOSTICS=true ;;
        --realistic)   REALISTIC=true ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

PASS_COUNT=0
FAIL_COUNT=0
BUILD_FAIL_COUNT=0

CXX_FLAGS="-std=c++17 -O2 -I.. -I/usr/include/eigen3"

# Optional feature flags
if $DIAGNOSTICS; then
    CXX_FLAGS="$CXX_FLAGS -DDIAGNOSTICS_ENABLED"
    echo "  Diagnostics: ENABLED"
fi
if $REALISTIC; then
    CXX_FLAGS="$CXX_FLAGS -DREALISTIC_DELAYS"
    echo "  Realistic delays: ENABLED"
fi

# acados setup
ACADOS_DIR="${ACADOS_INSTALL_DIR:-/home/stephen/acados}"
ACADOS_GEN_DIR="../acados_ocp/c_generated_code"

ACADOS_INCLUDES="-DACADOS_WITH_QPOASES -I$ACADOS_GEN_DIR -I$ACADOS_DIR/include -I$ACADOS_DIR/include/acados -I$ACADOS_DIR/include/blasfeo/include -I$ACADOS_DIR/include/hpipm/include"
ACADOS_LINK_LIBS="-L$ACADOS_DIR/lib -lacados -lhpipm -lblasfeo -lqpOASES_e -lm"
ACADOS_RPATH="-Wl,-rpath,$ACADOS_DIR/lib"
CXX_FLAGS="$CXX_FLAGS $ACADOS_INCLUDES $ACADOS_RPATH"

echo "*** acados MPCC solver ***"
echo "  ACADOS_DIR=$ACADOS_DIR"

# Pre-compile all acados C files into a static library (must use C compiler, not C++)
echo "  Building acados generated code..."
ACADOS_LIB_A="$SCRIPT_DIR/libacados_gen.a"
ACADOS_CC_FLAGS="-std=c99 -O2 $ACADOS_INCLUDES"
ACADOS_OBJS=""
for cfile in $ACADOS_GEN_DIR/acados_solver_mpcc_qcar2.c \
             $ACADOS_GEN_DIR/mpcc_qcar2_model/*.c \
             $ACADOS_GEN_DIR/mpcc_qcar2_cost/*.c \
             $ACADOS_GEN_DIR/mpcc_qcar2_constraints/*.c; do
    ofile="${cfile%.c}.o"
    gcc $ACADOS_CC_FLAGS -c -o "$ofile" "$cfile" 2>/dev/null
    ACADOS_OBJS="$ACADOS_OBJS $ofile"
done
ar rcs "$ACADOS_LIB_A" $ACADOS_OBJS 2>/dev/null
echo "  Built $ACADOS_LIB_A"
echo ""

build_and_run() {
    local name="$1"
    shift
    local srcs="$@"

    printf "%-35s " "$name"

    # Build
    if ! g++ $CXX_FLAGS -o "$name" $srcs $ACADOS_LIB_A $ACADOS_LINK_LIBS > "$OUTPUT_DIR/${name}_build.log" 2>&1; then
        printf "${RED}BUILD FAILED${NC}\n"
        BUILD_FAIL_COUNT=$((BUILD_FAIL_COUNT + 1))
        if $VERBOSE; then
            cat "$OUTPUT_DIR/${name}_build.log"
        fi
        return 1
    fi

    # Run
    if env LD_LIBRARY_PATH=$ACADOS_DIR/lib:${LD_LIBRARY_PATH:-} ./"$name" > "$OUTPUT_DIR/${name}.log" 2>&1; then
        local results=$(grep -o 'Results:.*' "$OUTPUT_DIR/${name}.log" | head -1)
        printf "${GREEN}PASS${NC}  %s\n" "$results"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        local results=$(grep -o 'Results:.*' "$OUTPUT_DIR/${name}.log" | head -1)
        local failures=$(grep 'FAIL' "$OUTPUT_DIR/${name}.log" | head -5)
        printf "${RED}FAIL${NC}  %s\n" "$results"
        if $VERBOSE; then
            echo "$failures"
        fi
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # Copy any CSV output generated (check both cwd and results/ subdir)
    for csv in *.csv; do
        [ -f "$csv" ] && mv "$csv" "$OUTPUT_DIR/" 2>/dev/null || true
    done
    for csv in results/*.csv; do
        [ -f "$csv" ] && mv "$csv" "$OUTPUT_DIR/" 2>/dev/null || true
    done

    return 0
}

echo "=============================================="
echo " MPCC Test Suite — $TIMESTAMP"
echo "=============================================="
echo "Output: $OUTPUT_DIR"
echo ""

ROAD_GRAPH="../road_graph.cpp"

# ---- Always run: Core solver tests ----
echo "--- Core Tests ---"
build_and_run "test_mpcc_solver" "test_mpcc_solver.cpp $ROAD_GRAPH"

if ! $QUICK; then
    # ---- Deployment tests (require road_graph) ----
    echo ""
    echo "--- Deployment Tests ---"
    build_and_run "test_deployment" "test_deployment.cpp $ROAD_GRAPH"
    build_and_run "test_full_mission_sim" "test_full_mission_sim.cpp $ROAD_GRAPH"
    build_and_run "test_mapframe_startup" "test_mapframe_startup.cpp $ROAD_GRAPH"

    # ---- Diagnostic tests ----
    echo ""
    echo "--- Diagnostics ---"
    build_and_run "diagnose_swerving" "diagnose_swerving.cpp $ROAD_GRAPH"
    build_and_run "diagnose_startup" "diagnose_startup.cpp $ROAD_GRAPH" || true
    build_and_run "diagnose_steering" "diagnose_steering.cpp $ROAD_GRAPH" || true

    # ---- Path tests ----
    echo ""
    echo "--- Path Tests ---"
    build_and_run "test_path_tangents" "test_path_tangents.cpp $ROAD_GRAPH" || true
fi

# ---- Summary ----
echo ""
echo "=============================================="
TOTAL=$((PASS_COUNT + FAIL_COUNT + BUILD_FAIL_COUNT))
printf " Summary: ${GREEN}%d passed${NC}, ${RED}%d failed${NC}, ${YELLOW}%d build errors${NC} / %d total\n" \
    "$PASS_COUNT" "$FAIL_COUNT" "$BUILD_FAIL_COUNT" "$TOTAL"
echo " Output:  $OUTPUT_DIR"
echo "=============================================="

# Generate plots from CSV if Python + matplotlib available
if command -v python3 &>/dev/null; then
    PLOT_SCRIPT="$SCRIPT_DIR/plot_results.py"
    if [ -f "$PLOT_SCRIPT" ]; then
        echo ""
        echo "Generating plots..."
        python3 "$PLOT_SCRIPT" "$OUTPUT_DIR" 2>/dev/null && \
            echo "Plots saved to $OUTPUT_DIR/*.png" || \
            echo "Plot generation failed (non-fatal)"
    fi
fi

# Exit with failure if any tests failed
if [ $FAIL_COUNT -gt 0 ] || [ $BUILD_FAIL_COUNT -gt 0 ]; then
    exit 1
fi
exit 0
