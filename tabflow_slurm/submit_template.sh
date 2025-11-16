#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=tabarena_run
#SBATCH --export=ALL
#SBATCH --requeue
#SBATCH --propagate=NONE
#SBATCH -o /work/dlclarge2/purucker-tabarena/slurm_out/new_runs/%A/slurm-%A_%a.out
#SBATCH -e /work/dlclarge2/purucker-tabarena/slurm_out/new_runs/%A/slurm-%A_%a.out

#### Set by outer script
## For GPU
##SBATCH --gres=gpu:1,localtmp:100
##SBATCH --partition=alldlc2_gpu-l40s # --partition=mldlc2_gpu-h200
## FOR CPU
##SBATCH --gres=localtmp:100
##SBATCH --partition=bosch_cpu-cascadelake
## For Both
##SBATCH --time=6:00:00
##SBATCH --mem-per-cpu=4G
##SBATCH --cpus-per-task=8

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
TABREPO_CACHE_DIR=$(jq -r '.defaults.tabrepo_cache_dir' "$JSON_FILE")
OUTPUT_DIR=$(jq -r '.defaults.output_dir' "$JSON_FILE")
NUM_CPUS=$(jq -r '.defaults.num_cpus' "$JSON_FILE")
NUM_GPUS=$(jq -r '.defaults.num_gpus' "$JSON_FILE")
MEMORY_LIMIT=$(jq -r '.defaults.memory_limit' "$JSON_FILE")
SETUP_RAY=$(jq -r '.defaults.setup_ray_for_slurm_shared_resources_environment' "$JSON_FILE")
IGNORE_CACHE=$(jq -r '.defaults.ignore_cache' "$JSON_FILE")
SEQUENTIAL_LOCAL_FOLD_FITTING=$(jq -r '.defaults.sequential_local_fold_fitting' "$JSON_FILE")

echo "Python Path: $PYTHON_PATH"
echo "Run Script: $RUNSCRIPT"
echo "OpenML Cache Directory: $OPENML_CACHE_DIR"
echo "Configs YAML File: $CONFIGS_YAML_FILE"
echo "Tabrepo Cache Directory: $TABREPO_CACHE_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of CPUs: $NUM_CPUS"
echo "Number of GPUs: $NUM_GPUS"
echo "Memory Limit: $MEMORY_LIMIT"
echo "Setup Ray for SLURM Shared Resources Environment: $SETUP_RAY"
echo "Ignore Cache: $IGNORE_CACHE"
echo "Sequential Local Fold Fitting: $SEQUENTIAL_LOCAL_FOLD_FITTING"

# Extract specific job fields
CONFIG_INDEX=$(jq -r --argjson J "$J" '.jobs[$J].config_index | join(",")' "$JSON_FILE")
TASK_ID=$(jq -r --argjson J "$J" '.jobs[$J].task_id' "$JSON_FILE")
FOLD=$(jq -r --argjson J "$J" '.jobs[$J].fold' "$JSON_FILE")
REPEAT=$(jq -r --argjson J "$J" '.jobs[$J].repeat' "$JSON_FILE")

# Output extracted values
echo "CONFIG_INDEX: $CONFIG_INDEX"
echo "Task ID: TASK_ID"
echo "Fold: $FOLD"
echo "Repeat: $REPEAT"



# Split CONFIG_INDEX into individual elements and loop
IFS=',' read -ra CONFIG_ARRAY <<< "$CONFIG_INDEX"

for CI in "${CONFIG_ARRAY[@]}"; do
    echo "Running config_index=$CI"

    $PYTHON_PATH $RUNSCRIPT \
        --task_id $TASK_ID \
        --fold $FOLD \
        --repeat $REPEAT \
        --config_index $CI \
        --configs_yaml_file $CONFIGS_YAML_FILE \
        --openml_cache_dir $OPENML_CACHE_DIR \
        --tabrepo_cache_dir $TABREPO_CACHE_DIR \
        --output_dir $OUTPUT_DIR \
        --num_cpus $NUM_CPUS \
        --num_gpus $NUM_GPUS \
        --memory_limit $MEMORY_LIMIT \
        --setup_ray_for_slurm_shared_resources_environment $SETUP_RAY \
        --ignore_cache $IGNORE_CACHE \
        --sequential_local_fold_fitting $SEQUENTIAL_LOCAL_FOLD_FITTING
done