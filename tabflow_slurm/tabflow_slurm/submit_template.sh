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
CONFIGS_YAML_FILE=$(jq -r '.defaults.configs_yaml_file' "$JSON_FILE")
OUTPUT_DIR=$(jq -r '.defaults.output_dir' "$JSON_FILE")
NUM_CPUS=$(jq -r '.defaults.num_cpus' "$JSON_FILE")
NUM_GPUS=$(jq -r '.defaults.num_gpus' "$JSON_FILE")
NUM_GPUS_MODEL=$(jq -r '.defaults.num_gpus_model' "$JSON_FILE")
MEMORY_LIMIT=$(jq -r '.defaults.memory_limit' "$JSON_FILE")
SETUP_RAY=$(jq -r '.defaults.setup_ray_for_slurm_shared_resources_environment' "$JSON_FILE")
IGNORE_CACHE=$(jq -r '.defaults.ignore_cache' "$JSON_FILE")
SEQUENTIAL_LOCAL_FOLD_FITTING=$(jq -r '.defaults.sequential_local_fold_fitting' "$JSON_FILE")
DYNAMIC_TABARENA_VALIDATION_PROTOCOL=$(
  jq -r '.defaults.dynamic_tabarena_validation_protocol // false' "$JSON_FILE"
)

echo "Python Path: $PYTHON_PATH"
echo "Run Script: $RUNSCRIPT"
echo "OpenML Cache Directory: $OPENML_CACHE_DIR"
echo "Configs YAML File: $CONFIGS_YAML_FILE"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of CPUs: $NUM_CPUS"
echo "Number of GPUs: $NUM_GPUS"
echo "Number of GPUs for model fitting: $NUM_GPUS_MODEL"
echo "Memory Limit: $MEMORY_LIMIT"
echo "Setup Ray for SLURM Shared Resources Environment: $SETUP_RAY"
echo "Ignore Cache: $IGNORE_CACHE"
echo "Sequential Local Fold Fitting: $SEQUENTIAL_LOCAL_FOLD_FITTING"
echo "Dynamic TabArena Validation Protocol: $DYNAMIC_TABARENA_VALIDATION_PROTOCOL"

run_one() {
    local TASK_ID="$1"
    local FOLD="$2"
    local REPEAT="$3"
    local CI="$4"
    echo "Running task_id=$TASK_ID fold=$FOLD repeat=$REPEAT config_index=$CI"

    $PYTHON_PATH $RUNSCRIPT \
        --task_id $TASK_ID \
        --fold $FOLD \
        --repeat $REPEAT \
        --config_index $CI \
        --configs_yaml_file $CONFIGS_YAML_FILE \
        --openml_cache_dir $OPENML_CACHE_DIR \
        --output_dir $OUTPUT_DIR \
        --num_cpus $NUM_CPUS \
        --num_gpus $NUM_GPUS \
        --num_gpus_model $NUM_GPUS_MODEL \
        --memory_limit $MEMORY_LIMIT \
        --setup_ray_for_slurm_shared_resources_environment $SETUP_RAY \
        --ignore_cache $IGNORE_CACHE \
        --sequential_local_fold_fitting $SEQUENTIAL_LOCAL_FOLD_FITTING \
        --dynamic_tabarena_validation_protocol $DYNAMIC_TABARENA_VALIDATION_PROTOCOL
}

# Bundle format: each job has `items: [...]` with one entry per
# (task_id, fold, repeat, config_index) tuple. Stream them as TSV
# (`task_id<TAB>fold<TAB>repeat<TAB>config_index`) so we don't depend on tuple
# indexing or argument quoting per item.
NUM_ITEMS=$(jq -r --argjson J "$J" '.jobs[$J].items | length' "$JSON_FILE")
echo "Bundle items: $NUM_ITEMS"

while IFS=$'\t' read -r TASK_ID FOLD REPEAT CI; do
    run_one "$TASK_ID" "$FOLD" "$REPEAT" "$CI"
done < <(jq -r --argjson J "$J" \
    '.jobs[$J].items[] | [.task_id, .fold, .repeat, .config_index] | @tsv' \
    "$JSON_FILE")