#!/usr/bin/env bash
# ============================================================================
# Nsight Profiling 完整工作流脚本
#
# 用法:
#   ./profile_workflow.sh step1   # nsys 时间线分析
#   ./profile_workflow.sh step2   # ncu 快速瓶颈定位
#   ./profile_workflow.sh step3   # ncu 详细分析
#   ./profile_workflow.sh step5   # ncu 前后对比
#   ./profile_workflow.sh all     # 运行完整流程 (步骤 1-3)
#   ./profile_workflow.sh clean   # 清理报告文件
#
# 环境: WSL2 Ubuntu + CUDA 13.x + RTX 3070 (sm_86)
# ============================================================================
set -euo pipefail

# ---- 配置 ----
PROF_DIR="$(cd "$(dirname "$0")" && pwd)"
REPORT_DIR="${PROF_DIR}/reports"
BINARY="${PROF_DIR}/example_kernel"
CUDA_ARCH="sm_86"
REPEAT=100

# Nsight 工具路径（WSL 内）
NSYS="$(which nsys 2>/dev/null || echo /usr/local/cuda/bin/nsys)"
NCU="$(which ncu 2>/dev/null || echo /usr/local/cuda/bin/ncu)"

# Windows 端 GUI 路径（用于查看报告）
NSYS_UI="/mnt/c/Program Files/NVIDIA Corporation/Nsight Systems 2025.1.3/host-windows-x64/nsys-ui.exe"
NCU_UI="/mnt/c/Program Files/NVIDIA Corporation/Nsight Compute 2025.2.1/host-windows-x64/ncu-ui.exe"

mkdir -p "${REPORT_DIR}"

# ---- 颜色输出 ----
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
step()  { echo -e "\n${CYAN}======== $* ========${NC}\n"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---- 编译 ----
compile() {
    step "编译 example_kernel (arch=${CUDA_ARCH})"
    nvcc -O2 -arch="${CUDA_ARCH}" -I/usr/local/cuda/include \
         -lnvidia-ml -o "${BINARY}" "${PROF_DIR}/example_kernel.cu"
    info "编译完成: ${BINARY}"
}

# ---- 步骤 1: nsys 时间线分析 ----
do_step1() {
    step "Step 1: nsys 时间线分析 — 找到最耗时的 kernel"

    compile

    local report="${REPORT_DIR}/nsys_timeline"
    info "运行 nsys profile..."

    "${NSYS}" profile \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --output="${report}" \
        --force-overwrite=true \
        "${BINARY}" "${REPEAT}"

    echo ""
    info "报告已生成: ${report}.qdrep (文本) / ${report}.nsys-rep (二进制)"
    echo ""
    info "=== Kernel 执行时间汇总 (按总时间降序) ==="
    "${NSYS}" stats --report cuda_gpu_kernsum "${report}.nsys-rep" 2>/dev/null || \
    "${NSYS}" stats --report cuda_gpu_kernsum "${report}.qdrep" 2>/dev/null || \
        warn "nsys stats 解析失败，请用 GUI 查看"

    echo ""
    info "=== GPU 内存使用汇总 ==="
    "${NSYS}" stats --report cuda_gpu_mem_size_sum "${report}.nsys-rep" 2>/dev/null || true

    echo ""
    info "查看 GUI（在 Windows 端）:"
    info "  方法1: PowerShell 执行 Start-Process '${NCU_UI}' -ArgumentList '${report}.nsys-rep'"
    info "  方法2: 直接在 Windows 资源管理器打开 \\\\wsl$\\Ubuntu${report}.nsys-rep"
}

# ---- 步骤 2: ncu 快速瓶颈定位 ----
do_step2() {
    step "Step 2: ncu 快速分析 — 定位瓶颈类型"

    compile

    local report="${REPORT_DIR}/ncu_quick"
    info "运行 ncu --set quick..."

    "${NCU}" \
        --set quick \
        --target-processes all \
        --launch-skip 5 \
        --launch-count 5 \
        -o "${report}" \
        "${BINARY}" "${REPEAT}"

    echo ""
    info "快速报告已生成: ${report}.ncu-rep"
    echo ""
    info "=== 关键指标摘要 ==="
    "${NCU}" --import "${report}.ncu-rep" \
        --page details \
        --metrics gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak,smsp__sass_average_data_bytes_per_sector_mem_global.pct,lts__t_sectors_pipe_lsu_mem_global_op_read.sum \
        --csv 2>/dev/null || true

    echo ""
    info "查看 GUI（在 Windows 端）:"
    info "  方法1: PowerShell 执行 Start-Process '${NCU_UI}' -ArgumentList '${report}.ncu-rep'"
    info "  方法2: 直接在 Windows 资源管理器打开 \\\\wsl$\\Ubuntu${report}.ncu-rep"
}

# ---- 步骤 3: ncu 详细分析 ----
do_step3() {
    step "Step 3: ncu 详细分析 — 逐项查看微架构指标"

    compile

    local report="${REPORT_DIR}/ncu_full"
    info "运行 ncu --set full（这会比较慢）..."

    "${NCU}" \
        --set full \
        --target-processes all \
        --launch-skip 5 \
        --launch-count 2 \
        -o "${report}" \
        "${BINARY}" "${REPEAT}"

    echo ""
    info "详细报告已生成: ${report}.ncu-rep"

    echo ""
    info "=== Occupancy 分析 ==="
    "${NCU}" --import "${report}.ncu-rep" \
        --page details \
        --section Occupancy \
        --csv 2>/dev/null || true

    echo ""
    info "=== 内存工作负载分析 ==="
    "${NCU}" --import "${report}.ncu-rep" \
        --page details \
        --section MemoryWorkloadAnalysis \
        --csv 2>/dev/null || true

    echo ""
    info "=== 内存带宽分析 ==="
    "${NCU}" --import "${report}.ncu-rep" \
        --page details \
        --section Memory \
        --csv 2>/dev/null || true

    echo ""
    info "=== Launch 统计 ==="
    "${NCU}" --import "${report}.ncu-rep" \
        --page details \
        --section LaunchStats \
        --csv 2>/dev/null || true

    echo ""
    info "=== Warp Stall 分析 ==="
    "${NCU}" --import "${report}.ncu-rep" \
        --page details \
        --section WarpStateStatistics \
        --csv 2>/dev/null || true

    echo ""
    info "查看完整 GUI 报告（在 Windows 端）:"
    info "  PowerShell: Start-Process '${NCU_UI}' -ArgumentList '${report}.ncu-rep'"
}

# ---- 步骤 5: 前后对比 ----
do_step5() {
    step "Step 5: ncu 前后对比 — 验证优化效果"

    local before="${1:-${REPORT_DIR}/ncu_before.ncu-rep}"
    local after="${2:-${REPORT_DIR}/ncu_after.ncu-rep}"

    if [[ ! -f "${before}" ]]; then
        info "before 报告不存在，先生成..."
        compile
        "${NCU}" \
            --set full \
            --launch-skip 5 \
            --launch-count 2 \
            -o "${REPORT_DIR}/ncu_before" \
            "${BINARY}" "${REPEAT}"
        before="${REPORT_DIR}/ncu_before.ncu-rep"
    fi

    if [[ ! -f "${after}" ]]; then
        warn "after 报告不存在: ${after}"
        info "请先优化代码，然后运行:"
        info "  ${NCU} --set full --launch-skip 5 --launch-count 2 -o ${REPORT_DIR}/ncu_after ${BINARY} ${REPEAT}"
        info "再重新执行: $0 step5 ${before} ${REPORT_DIR}/ncu_after.ncu-rep"
        return 1
    fi

    echo ""
    info "=== 对比: GPU 时间 ==="
    "${NCU}" --import "${before}" "${after}" \
        --page details \
        --metrics gpu__time_duration.sum \
        --csv 2>/dev/null || true

    echo ""
    info "=== 对比: Occupancy ==="
    "${NCU}" --import "${before}" "${after}" \
        --page details \
        --metrics sm__warps_active.avg.pct_of_peak \
        --csv 2>/dev/null || true

    echo ""
    info "=== 对比: 内存吞吐 ==="
    "${NCU}" --import "${before}" "${after}" \
        --page details \
        --metrics lts__t_sectors_pipe_lsu_mem_global_op_read.sum,lts__t_sectors_pipe_lsu_mem_global_op_write.sum \
        --csv 2>/dev/null || true

    echo ""
    info "GUI 并排对比（在 Windows 端）:"
    info "  PowerShell: Start-Process '${NCU_UI}' -ArgumentList '${before}','${after}'"
}

# ---- 清理 ----
do_clean() {
    step "清理报告文件"
    rm -rf "${REPORT_DIR}"
    rm -f "${BINARY}"
    info "已清理"
}

# ---- 主入口 ----
usage() {
    echo "用法: $0 <command> [args...]"
    echo ""
    echo "命令:"
    echo "  step1       步骤1: nsys 时间线分析（找最耗时 kernel）"
    echo "  step2       步骤2: ncu 快速分析（定位瓶颈类型）"
    echo "  step3       步骤3: ncu 详细分析（微架构指标）"
    echo "  step5       步骤5: ncu 前后对比（验证优化）"
    echo "              step5 <before.ncu-rep> <after.ncu-rep>"
    echo "  all         运行步骤 1-3"
    echo "  compile     仅编译"
    echo "  clean       清理报告和二进制"
    echo ""
    echo "工作流: step1 → step2 → step3 → 优化代码 → step5 → 回到 step1"
}

case "${1:-}" in
    step1)   do_step1 ;;
    step2)   do_step2 ;;
    step3)   do_step3 ;;
    step5)   do_step5 "${2:-}" "${3:-}" ;;
    all)     do_step1; do_step2; do_step3 ;;
    compile) compile ;;
    clean)   do_clean ;;
    *)       usage ;;
esac
