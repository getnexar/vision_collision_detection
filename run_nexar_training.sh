#!/bin/bash

# Nexar Video Classification - Distributed Training Runner
# This script provides easy commands to run your Nexar training in different modes

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="nexar_train_distributed.py"

# Default parameters (matching your notebook)
BASE_DIRS="${BASE_DIRS:-../data/research-nvidia-data/nvidia-1,../data/research-nvidia-data/nvidia-2}"
METADATA_CSV="${METADATA_CSV:-nvidia_delivery_to_train.csv}"
SAVE_DIR="${SAVE_DIR:-model_results}"

BASE_MODEL="${BASE_MODEL:-convnext_tiny}"
TEMPORAL_MODE="${TEMPORAL_MODE:-gru}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    print_info "Checking requirements..."
    
    # Check if Python script exists
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "Python script $PYTHON_SCRIPT not found!"
        exit 1
    fi
    
    # Check if data directories exist
    IFS=',' read -ra DIRS <<< "$BASE_DIRS"
    for dir in "${DIRS[@]}"; do
        if [ ! -d "$dir" ]; then
            print_warning "Data directory $dir not found"
        fi
    done
    
    # Check if metadata CSV exists
    if [ ! -f "$METADATA_CSV" ]; then
        print_warning "Metadata CSV $METADATA_CSV not found"
    fi
    
    # Check if required Python modules can be imported
    python -c "import torch; import nexar_train; import nexar_videos" 2>/dev/null || {
        print_error "Required Python modules not found. Make sure nexar_train.py and nexar_videos.py are available."
        exit 1
    }
    
    print_info "Requirements check completed"
}

print_gpu_info() {
    print_info "GPU Information:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | nl -v0 -s": "
    else
        print_warning "nvidia-smi not available"
    fi
}

run_single_gpu() {
    print_header "SINGLE GPU TRAINING (Like your notebook)"
    
    check_requirements
    print_gpu_info
    
    print_info "Starting single GPU training..."
    print_info "Base Model: $BASE_MODEL"
    print_info "Temporal Mode: $TEMPORAL_MODE"
    print_info "Epochs: $EPOCHS"
    print_info "Batch Size: $BATCH_SIZE"
    
    python "$PYTHON_SCRIPT" \
        --base-dirs ${BASE_DIRS//,/ } \
        --metadata-csv "$METADATA_CSV" \
        --base-model "$BASE_MODEL" \
        --temporal-mode "$TEMPORAL_MODE" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --save-dir "$SAVE_DIR" \
        --num-workers "$NUM_WORKERS" \
        --use-class-weights \
        "$@"
}

run_distributed() {
    local num_gpus=${1:-8}
    
    print_header "DISTRIBUTED TRAINING ($num_gpus GPUs)"
    
    check_requirements
    print_gpu_info
    
    # Check if we have enough GPUs
    local available_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$available_gpus" -lt "$num_gpus" ]; then
        print_warning "Requested $num_gpus GPUs but only $available_gpus available"
        print_info "Adjusting to use $available_gpus GPUs"
        num_gpus=$available_gpus
    fi
    
    print_info "Starting distributed training on $num_gpus GPUs..."
    print_info "Base Model: $BASE_MODEL"
    print_info "Temporal Mode: $TEMPORAL_MODE"
    print_info "Epochs: $EPOCHS"
    print_info "Batch Size per GPU: $BATCH_SIZE"
    print_info "Effective Batch Size: $((BATCH_SIZE * num_gpus))"
    
    torchrun --nproc_per_node="$num_gpus" "$PYTHON_SCRIPT" \
        --base-dirs ${BASE_DIRS//,/ } \
        --metadata-csv "$METADATA_CSV" \
        --base-model "$BASE_MODEL" \
        --temporal-mode "$TEMPORAL_MODE" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --save-dir "$SAVE_DIR" \
        --num-workers "$NUM_WORKERS" \
        --use-class-weights \
        "${@:2}"
}

run_grid_search() {
    local num_gpus=${1:-1}
    
    print_header "GRID SEARCH ($num_gpus GPUs)"
    
    check_requirements
    
    if [ "$num_gpus" -eq 1 ]; then
        print_info "Running grid search on single GPU..."
        python "$PYTHON_SCRIPT" \
            --base-dirs ${BASE_DIRS//,/ } \
            --metadata-csv "$METADATA_CSV" \
            --save-dir "$SAVE_DIR" \
            --run-grid-search \
            "${@:2}"
    else
        print_info "Running grid search on $num_gpus GPUs..."
        torchrun --nproc_per_node="$num_gpus" "$PYTHON_SCRIPT" \
            --base-dirs ${BASE_DIRS//,/ } \
            --metadata-csv "$METADATA_CSV" \
            --save-dir "$SAVE_DIR" \
            --run-grid-search \
            "${@:2}"
    fi
}

run_quick_test() {
    print_header "QUICK TEST (1 epoch)"
    
    print_info "Running quick test with 1 epoch..."
    
    python "$PYTHON_SCRIPT" \
        --base-dirs ${BASE_DIRS//,/ } \
        --metadata-csv "$METADATA_CSV" \
        --base-model "$BASE_MODEL" \
        --temporal-mode "$TEMPORAL_MODE" \
        --epochs 1 \
        --batch-size "$BATCH_SIZE" \
        --save-dir "${SAVE_DIR}_test" \
        --experiment-name "quick_test" \
        "$@"
}

show_usage() {
    cat << EOF
Nexar Video Classification - Distributed Training Runner

Usage: $0 <command> [options]

Commands:
    single                  Run single GPU training (like your notebook)
    distributed [num_gpus]  Run distributed training (default: 8 GPUs)
    grid-search [num_gpus]  Run grid search experiments
    test                    Run quick test (1 epoch)
    check                   Check requirements only
    help                    Show this help message

Environment Variables:
    BASE_DIRS              Data directories (comma-separated)
    METADATA_CSV           Metadata CSV file path
    BASE_MODEL             Model architecture (default: convnext_tiny)
    TEMPORAL_MODE          Temporal mode (default: gru)
    EPOCHS                 Number of epochs (default: 15)
    BATCH_SIZE             Batch size per GPU (default: 8)
    LEARNING_RATE          Learning rate (default: 1e-4)
    SAVE_DIR               Save directory (default: model_results)

Examples:
    # Single GPU (exactly like your notebook):
    $0 single

    # 8 GPUs distributed:
    $0 distributed 8

    # 4 GPUs with custom model:
    BASE_MODEL=resnet18 TEMPORAL_MODE=attention $0 distributed 4

    # Grid search on single GPU:
    $0 grid-search 1

    # Quick test:
    $0 test

    # Custom parameters:
    $0 single --epochs 30 --batch-size 16 --learning-rate 5e-5

EOF
}

# Main script logic
case "${1:-help}" in
    single)
        shift
        run_single_gpu "$@"
        ;;
    distributed)
        shift
        run_distributed "$@"
        ;;
    grid-search)
        shift
        run_grid_search "$@"
        ;;
    test)
        shift
        run_quick_test "$@"
        ;;
    check)
        check_requirements
        print_gpu_info
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        echo
        show_usage
        exit 1
        ;;
esac