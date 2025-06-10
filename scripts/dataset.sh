#!/bin/bash

# Single GPU + 6 CPU Recipe Processing Script
# This script runs the recipe processing with optimal settings for 1 GPU and 6 CPU cores

set -e  # Exit on any error

# Configuration
DATA_PATH="${1:-/Users/connectednorth/Documents/Remy.ai/data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv}"
MODEL_NAME="${2:-google/flan-t5-base}"
BATCH_SIZE="${3:-16}"
OUTPUT_PATH="${4:-cleaned_recipes_single_gpu.csv}"
NUM_CPU_WORKERS="${5:-6}"
STRATEGY="${6:-hybrid}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if GPU is available
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_count=$(nvidia-smi --list-gpus | wc -l)
        if [ "$gpu_count" -gt 0 ]; then
            print_success "Found $gpu_count GPU(s)"
            nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
            return 0
        else
            print_warning "nvidia-smi found but no GPUs detected"
            return 1
        fi
    else
        print_warning "nvidia-smi not found - GPU may not be available"
        return 1
    fi
}

# Function to check Python environment
check_python_env() {
    print_status "Checking Python environment..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 not found"
        exit 1
    fi
    
    # Check if required packages are installed
    local required_packages=("torch" "transformers" "pandas" "accelerate")
    local missing_packages=()
    
    for package in "${required_packages[@]}"; do
        if ! python3 -c "import $package" &> /dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        print_error "Missing required packages: ${missing_packages[*]}"
        print_status "Install with: pip install ${missing_packages[*]}"
        exit 1
    fi
    
    print_success "All required packages are installed"
}

# Function to check if data file exists
check_data_file() {
    if [ ! -f "$DATA_PATH" ]; then
        print_error "Data file not found: $DATA_PATH"
        print_status "Please provide the correct path as the first argument"
        exit 1
    fi
    
    # Check file size
    local file_size=$(du -h "$DATA_PATH" | cut -f1)
    print_success "Data file found: $DATA_PATH (Size: $file_size)"
    
    # Count lines in CSV (rough estimate of recipes)
    local line_count=$(wc -l < "$DATA_PATH")
    print_status "Estimated recipes: $((line_count - 1))"
}

# Function to set optimal environment variables
set_environment() {
    print_status "Setting up environment variables..."
    
    # Set number of CPU threads
    export OMP_NUM_THREADS=$NUM_CPU_WORKERS
    export MKL_NUM_THREADS=$NUM_CPU_WORKERS
    export NUMEXPR_NUM_THREADS=$NUM_CPU_WORKERS
    
    # PyTorch settings
    export TORCH_NUM_THREADS=$NUM_CPU_WORKERS
    
    # Disable tokenizers parallelism to avoid warnings
    export TOKENIZERS_PARALLELISM=false
    
    # CUDA settings
    export CUDA_VISIBLE_DEVICES=0  # Use only the first GPU
    
    print_success "Environment configured for 1 GPU and $NUM_CPU_WORKERS CPU cores"
}

# Function to estimate processing time
estimate_time() {
    local line_count=$(wc -l < "$DATA_PATH")
    local recipe_count=$((line_count - 1))
    
    # Rough estimates based on batch size and hardware
    local batches_per_second=2  # Conservative estimate for T5-base
    local total_batches=$((recipe_count / BATCH_SIZE + 1))
    local estimated_seconds=$((total_batches / batches_per_second))
    local estimated_minutes=$((estimated_seconds / 60))
    
    print_status "Estimated processing time: ~$estimated_minutes minutes for $recipe_count recipes"
}

# Function to run the processing
run_processing() {
    print_status "Starting recipe processing..."
    print_status "Configuration:"
    echo "  - Data Path: $DATA_PATH"
    echo "  - Model: $MODEL_NAME"
    echo "  - Batch Size: $BATCH_SIZE"
    echo "  - Strategy: $STRATEGY"
    echo "  - CPU Workers: $NUM_CPU_WORKERS"
    echo "  - Output: $OUTPUT_PATH"
    echo ""
    
    # Record start time
    local start_time=$(date +%s)
    
    # Run the processing based on strategy
    case $STRATEGY in
        "accelerate")
            print_status "Using Accelerate strategy (single GPU)..."
            python3 recipe_processor.py \
                --data_path "$DATA_PATH" \
                --model_name "$MODEL_NAME" \
                --batch_size "$BATCH_SIZE" \
                --strategy accelerate \
                --output_path "$OUTPUT_PATH"
            ;;
        "hybrid")
            print_status "Using Hybrid CPU-GPU strategy..."
            python3 recipe_processor.py \
                --data_path "$DATA_PATH" \
                --model_name "$MODEL_NAME" \
                --batch_size "$BATCH_SIZE" \
                --strategy hybrid \
                --num_cpu_workers "$NUM_CPU_WORKERS" \
                --output_path "$OUTPUT_PATH"
            ;;
        *)
            print_error "Invalid strategy: $STRATEGY"
            print_status "Valid strategies: accelerate, hybrid"
            exit 1
            ;;
    esac
    
    # Calculate processing time
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_minutes=$((duration / 60))
    local duration_seconds=$((duration % 60))
    
    print_success "Processing completed in ${duration_minutes}m ${duration_seconds}s"
}

# Function to validate output
validate_output() {
    if [ -f "$OUTPUT_PATH" ]; then
        local output_lines=$(wc -l < "$OUTPUT_PATH")
        local input_lines=$(wc -l < "$DATA_PATH")
        
        if [ "$output_lines" -eq "$input_lines" ]; then
            print_success "Output file created successfully: $OUTPUT_PATH"
            print_status "Records processed: $((output_lines - 1))"
            
            # Show file size
            local file_size=$(du -h "$OUTPUT_PATH" | cut -f1)
            print_status "Output file size: $file_size"
        else
            print_warning "Output line count ($output_lines) doesn't match input ($input_lines)"
        fi
    else
        print_error "Output file not created: $OUTPUT_PATH"
    fi
}

# Function to show GPU memory usage
show_gpu_memory() {
    if command -v nvidia-smi &> /dev/null; then
        print_status "GPU Memory Usage:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | \
        awk '{printf "  Used: %d MB / %d MB (%.1f%%)\n", $1, $2, ($1/$2)*100}'
    fi
}

# Function to cleanup on exit
cleanup() {
    print_status "Cleaning up..."
    # Kill any remaining Python processes (optional)
    # pkill -f "recipe_processor.py" 2>/dev/null || true
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [DATA_PATH] [MODEL_NAME] [BATCH_SIZE] [OUTPUT_PATH] [NUM_CPU_WORKERS] [STRATEGY]"
    echo ""
    echo "Arguments:"
    echo "  DATA_PATH         Path to the recipe CSV file"
    echo "  MODEL_NAME        Hugging Face model name (default: google/flan-t5-base)"
    echo "  BATCH_SIZE        Batch size for processing (default: 16)"
    echo "  OUTPUT_PATH       Output CSV file path (default: cleaned_recipes_single_gpu.csv)"
    echo "  NUM_CPU_WORKERS   Number of CPU workers (default: 6)"
    echo "  STRATEGY          Processing strategy: accelerate|hybrid (default: hybrid)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/recipes.csv"
    echo "  $0 /path/to/recipes.csv google/flan-t5-small 32"
    echo "  $0 /path/to/recipes.csv google/flan-t5-base 16 output.csv 8 hybrid"
}

# Main execution
main() {
    print_status "=== Single GPU Recipe Processing Script ==="
    echo ""
    
    # Check for help flag
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        show_usage
        exit 0
    fi
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Run checks
    check_python_env
    check_gpu
    check_data_file
    
    # Setup environment
    set_environment
    
    # Show estimation
    estimate_time
    
    # Show initial GPU memory
    show_gpu_memory
    
    echo ""
    print_status "Press Ctrl+C to cancel, or wait 5 seconds to continue..."
    sleep 5
    
    # Run processing
    run_processing
    
    # Validate output
    validate_output
    
    # Show final GPU memory
    echo ""
    show_gpu_memory
    
    print_success "Recipe processing completed successfully!"
}

# Run main function
main "$@"