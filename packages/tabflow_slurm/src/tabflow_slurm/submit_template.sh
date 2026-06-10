#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=tabarena_run
#SBATCH --export=ALL,TABPFN_DISABLE_TELEMETRY=1,PYTHONUNBUFFERED=1
#SBATCH --requeue
#SBATCH --propagate=NONE

set -e
set -u
set -o pipefail
set -x

# Ensure jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install it with 'sudo apt install jq' or 'brew install jq'."
    exit 1
fi

# File path to JSON
JSON_FILE=${1:?Error: JSON file argument is required}
echo "Using JSON file: $JSON_FILE"
# Select job index from arguments
J=${SLURM_ARRAY_TASK_ID}  # Default to index 0 if not provided
echo "Selected Job Index: $J"

# Read defaults
PYTHON_PATH=$(jq -r '.defaults.python' "$JSON_FILE")
RUNSCRIPT=$(jq -r '.defaults.run_script' "$JSON_FILE")
OPENML_CACHE_DIR=$(jq -r '.defaults.openml_cache_dir' "$JSON_FILE")
JOB_BATCH_DIR=$(jq -r '.defaults.job_batch_dir' "$JSON_FILE")
OUTPUT_DIR=$(jq -r '.defaults.output_dir' "$JSON_FILE")
NUM_CPUS=$(jq -r '.defaults.num_cpus' "$JSON_FILE")
NUM_GPUS=$(jq -r '.defaults.num_gpus' "$JSON_FILE")
MEMORY_LIMIT=$(jq -r '.defaults.memory_limit' "$JSON_FILE")
SETUP_RAY=$(jq -r '.defaults.setup_ray_for_slurm_shared_resources_environment' "$JSON_FILE")
IGNORE_CACHE=$(jq -r '.defaults.ignore_cache' "$JSON_FILE")

echo "Python Path: $PYTHON_PATH"
echo "Run Script: $RUNSCRIPT"
echo "OpenML Cache Directory: $OPENML_CACHE_DIR"
echo "Job Batch Dir: $JOB_BATCH_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of CPUs: $NUM_CPUS"
echo "Number of GPUs: $NUM_GPUS"
echo "Memory Limit: $MEMORY_LIMIT"
echo "Setup Ray for SLURM Shared Resources Environment: $SETUP_RAY"
echo "Ignore Cache: $IGNORE_CACHE"

run_one() {
    local EXPERIMENT="$1"
    local DATASET="$2"
    local FOLD="$3"
    local REPEAT="$4"
    echo "Running experiment=$EXPERIMENT dataset=$DATASET fold=$FOLD repeat=$REPEAT"

    $PYTHON_PATH $RUNSCRIPT \
        --experiment "$EXPERIMENT" \
        --dataset "$DATASET" \
        --fold $FOLD \
        --repeat $REPEAT \
        --job_batch_dir "$JOB_BATCH_DIR" \
        --openml_cache_dir $OPENML_CACHE_DIR \
        --output_dir $OUTPUT_DIR \
        --num_cpus $NUM_CPUS \
        --num_gpus $NUM_GPUS \
        --memory_limit $MEMORY_LIMIT \
        --setup_ray_for_slurm_shared_resources_environment $SETUP_RAY \
        --ignore_cache $IGNORE_CACHE
}

# Bundle format: each job has `items: [...]` with one entry per
# (experiment, dataset, fold, repeat) work unit. Stream them as TSV
# (`experiment<TAB>dataset<TAB>fold<TAB>repeat`) so we don't depend on tuple
# indexing or argument quoting per item.
NUM_ITEMS=$(jq -r --argjson J "$J" '.jobs[$J].items | length' "$JSON_FILE")
echo "Bundle items: $NUM_ITEMS"

while IFS=$'\t' read -r EXPERIMENT DATASET FOLD REPEAT; do
    run_one "$EXPERIMENT" "$DATASET" "$FOLD" "$REPEAT"
done < <(jq -r --argjson J "$J" \
    '.jobs[$J].items[] | [.experiment, .dataset, .fold, .repeat] | @tsv' \
    "$JSON_FILE")