#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --job-name=tabarena_run
#SBATCH --export=ALL,TABPFN_DISABLE_TELEMETRY=1,HF_HUB_DISABLE_PROGRESS_BARS=1
#SBATCH --requeue
#SBATCH --propagate=NONE

# One SLURM array task: runs every item of jobs[SLURM_ARRAY_TASK_ID] from the job JSON, one
# python process per item, inside a per-job node-local scratch directory that is removed on exit.
# A failing item does not abort its siblings; the task still exits non-zero so sacct shows FAILED
# and the log carries one "##### item FAILED" line per failed item.

set -e
set -u
set -o pipefail
set -x

if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install it with 'sudo apt install jq' or 'brew install jq'."
    exit 1
fi

# Paths are resolved to absolute form before we cd into the scratch dir. The interpreter path is
# kept as written: resolving the venv symlink would land on the system python outside the venv.
JSON_FILE=$(readlink -f "${1:?Error: JSON file argument is required}")
echo "Using JSON file: $JSON_FILE"
J=${SLURM_ARRAY_TASK_ID}
echo "Selected Job Index: $J"

PYTHON_PATH=$(jq -r '.defaults.python' "$JSON_FILE")
RUNSCRIPT=$(readlink -f "$(jq -r '.defaults.run_script' "$JSON_FILE")")
JOB_BATCH_DIR=$(readlink -m "$(jq -r '.defaults.job_batch_dir' "$JSON_FILE")")
OUTPUT_DIR=$(readlink -m "$(jq -r '.defaults.output_dir' "$JSON_FILE")")
NUM_CPUS=$(jq -r '.defaults.num_cpus' "$JSON_FILE")
NUM_GPUS=$(jq -r '.defaults.num_gpus' "$JSON_FILE")
MEMORY_LIMIT=$(jq -r '.defaults.memory_limit' "$JSON_FILE")
SETUP_RAY=$(jq -r '.defaults.setup_ray_for_slurm_shared_resources_environment' "$JSON_FILE")
IGNORE_CACHE=$(jq -r '.defaults.ignore_cache' "$JSON_FILE")
# Keys below are optional so job JSONs written before they existed keep working.
OFFLINE_WEIGHTS=$(jq -r '.defaults.offline_weights // false | tostring' "$JSON_FILE")
REQUIRE_WARMUP=$(jq -r '.defaults.require_warmup // true | tostring' "$JSON_FILE")
SLURM_LOG_DIR=$(jq -r '.defaults.slurm_log_dir // empty' "$JSON_FILE")
STAGE_WEIGHTS=$(jq -r '.defaults.staging.stage_weights // false | tostring' "$JSON_FILE")
PRETOUCH_LIBS=$(jq -r '.defaults.staging.pretouch_libs // false | tostring' "$JSON_FILE")
RESERVE_BYTES=$(jq -r '.defaults.staging.reserve_bytes // 10737418240' "$JSON_FILE")
PRETOUCH_MAX_BYTES=$(jq -r '.defaults.staging.pretouch_max_bytes // 2147483648' "$JSON_FILE")
JIT_CACHE_MAX_MB=$(jq -r '.defaults.staging.jit_cache_max_mb // 2048' "$JSON_FILE")

echo "Python Path: $PYTHON_PATH"
echo "Run Script: $RUNSCRIPT"
echo "Job Batch Dir: $JOB_BATCH_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of CPUs: $NUM_CPUS"
echo "Number of GPUs: $NUM_GPUS"
echo "Memory Limit: $MEMORY_LIMIT"
echo "Setup Ray for SLURM Shared Resources Environment: $SETUP_RAY"
echo "Ignore Cache: $IGNORE_CACHE"
echo "Offline Weights: $OFFLINE_WEIGHTS"
echo "Require Warm-up: $REQUIRE_WARMUP"
echo "Stage Weights: $STAGE_WEIGHTS  Pretouch Libs: $PRETOUCH_LIBS"

# Environment hygiene: thread pools are sized by the models (and the runner's affinity check),
# never by a variable inherited from the submitting shell. PYTORCH_CUDA_ALLOC_CONF is left alone
# (an operator value is a documented contract of the Mitra-v2 wrapper).
for v in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_MAX_THREADS OMP_PROC_BIND OMP_PLACES; do
    if [ -n "${!v:-}" ]; then echo "hygiene: unsetting $v=${!v}"; fi
    unset "$v"
done

# Per-job node-local scratch. Short name so Ray's AF_UNIX socket paths stay under 107 bytes.
SCRATCH_ROOT="${TMPDIR:-/tmp}"
mkdir -p "$SCRATCH_ROOT"
JOB_SCRATCH="$SCRATCH_ROOT/tj_${SLURM_JOB_ID}"
mkdir -p "$JOB_SCRATCH/tmp" "$JOB_SCRATCH/ag" "$JOB_SCRATCH/stage" "$JOB_SCRATCH/ray"
export TMPDIR="$JOB_SCRATCH/tmp"
# AGWrapper roots this fit's predictor artifacts here without touching the shipped experiments.
export TABARENA_MODEL_ARTIFACTS_BASE_PATH="$JOB_SCRATCH/ag"
if [ -n "$SLURM_LOG_DIR" ]; then
    # Ray worker logs of a failed item land next to this array job's SLURM output files.
    export TABARENA_RAY_LOG_DIR="$SLURM_LOG_DIR/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}/ray_logs/task_${J}"
fi

cleanup_job_scratch() {
    local rc=$?
    set +e
    cd /
    rm -rf "$JOB_SCRATCH"
    exit "$rc"
}
trap cleanup_job_scratch EXIT
trap 'exit 143' TERM

# JIT caches (numba, Triton, NVIDIA) keyed per node, user and venv so later jobs on a warm node
# reuse them; numba and Triton write atomically and validate source stamps, so sharing is safe.
# The cap bounds growth on the node's disk.
VENV_ID="${USER:-$(id -un)}_$(basename "$(dirname "$(dirname "$PYTHON_PATH")")")_$(printf '%s' "$PYTHON_PATH" | cksum | cut -d' ' -f1)"
JIT_CACHE_ROOT="${TABARENA_JIT_ROOT:-$SCRATCH_ROOT/tabarena_jit}/$VENV_ID"
if [ -d "$JIT_CACHE_ROOT" ]; then
    JIT_MB=$(du -sm "$JIT_CACHE_ROOT" 2>/dev/null | cut -f1 || echo 0)
    if [[ "$JIT_MB" =~ ^[0-9]+$ ]] && [ "$JIT_MB" -gt "$JIT_CACHE_MAX_MB" ]; then
        echo "jit: cache $JIT_CACHE_ROOT is ${JIT_MB} MB (cap ${JIT_CACHE_MAX_MB} MB), clearing it"
        rm -rf "$JIT_CACHE_ROOT"
    fi
fi
export NUMBA_CACHE_DIR="$JIT_CACHE_ROOT/numba"
export TRITON_CACHE_DIR="$JIT_CACHE_ROOT/triton"
export CUDA_CACHE_PATH="$JIT_CACHE_ROOT/nv"
mkdir -p "$NUMBA_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

# Ray forwards the driver's cwd into every worker's sys.path, so leave the submission directory.
cd "$JOB_SCRATCH"
df -h "$SCRATCH_ROOT" || true

# Optional node-local copy of the run's foundation-model weights (defaults.staging, written by the
# setup). Every step fails closed: on any error the function returns non-zero, no cache variable is
# exported, and the fit reads the shared filesystem as before. Non-staged repo dirs and checkpoints
# of the source caches are overlaid as symlinks so unenumerated assets still resolve.
stage_weights() {
    local hub_src tabpfn_src need avail entry name
    command -v rsync >/dev/null 2>&1 || { echo "staging: rsync not found"; return 1; }
    hub_src=$(jq -r '.defaults.staging.hf_hub_cache_src // empty' "$JSON_FILE") || return 1
    tabpfn_src=$(jq -r '.defaults.staging.tabpfn_cache_dir_src // empty' "$JSON_FILE") || return 1
    local -a hf_dirs tabpfn_files
    mapfile -t hf_dirs < <(jq -r '.defaults.staging.hf_repo_dirs[]?' "$JSON_FILE") || return 1
    mapfile -t tabpfn_files < <(jq -r '.defaults.staging.tabpfn_files[]?' "$JSON_FILE") || return 1
    if [ "${#hf_dirs[@]}" -eq 0 ] && [ "${#tabpfn_files[@]}" -eq 0 ]; then
        echo "staging: nothing to stage"; return 1
    fi
    for entry in "${hf_dirs[@]}" "${tabpfn_files[@]}"; do
        [ -e "$entry" ] || { echo "staging: source missing: $entry"; return 1; }
    done
    need=$(du -sbc "${hf_dirs[@]}" "${tabpfn_files[@]}" 2>/dev/null | tail -1 | cut -f1) || return 1
    avail=$(df --output=avail -B1 "$JOB_SCRATCH" 2>/dev/null | tail -1 | tr -d ' ') || return 1
    [[ "$need" =~ ^[0-9]+$ ]] && [[ "$avail" =~ ^[0-9]+$ ]] || { echo "staging: could not size the copy"; return 1; }
    if [ "$avail" -lt $(( need + RESERVE_BYTES )) ]; then
        echo "staging: skipped, need $need + reserve $RESERVE_BYTES > avail $avail bytes"; return 1
    fi
    if [ "${#hf_dirs[@]}" -gt 0 ]; then
        mkdir -p "$JOB_SCRATCH/stage/huggingface/hub" || return 1
        rsync -a "${hf_dirs[@]}" "$JOB_SCRATCH/stage/huggingface/hub/" || return 1
        if [ -n "$hub_src" ] && [ -d "$hub_src" ]; then
            for entry in "$hub_src"/models--*; do
                [ -d "$entry" ] || continue
                name=$(basename "$entry")
                [ -e "$JOB_SCRATCH/stage/huggingface/hub/$name" ] || ln -s "$entry" "$JOB_SCRATCH/stage/huggingface/hub/$name" || return 1
            done
        fi
        touch "$JOB_SCRATCH/stage/.hf_ok" || return 1
    fi
    if [ "${#tabpfn_files[@]}" -gt 0 ]; then
        mkdir -p "$JOB_SCRATCH/stage/tabpfn" || return 1
        rsync -a "${tabpfn_files[@]}" "$JOB_SCRATCH/stage/tabpfn/" || return 1
        if [ -n "$tabpfn_src" ] && [ -d "$tabpfn_src" ]; then
            for entry in "$tabpfn_src"/*.ckpt "$tabpfn_src"/*.cpkt; do
                [ -f "$entry" ] || continue
                name=$(basename "$entry")
                [ -e "$JOB_SCRATCH/stage/tabpfn/$name" ] || ln -s "$entry" "$JOB_SCRATCH/stage/tabpfn/$name" || return 1
            done
        fi
        touch "$JOB_SCRATCH/stage/.tabpfn_ok" || return 1
    fi
    echo "staging: staged $need bytes into $JOB_SCRATCH/stage"
    return 0
}
if [ "$STAGE_WEIGHTS" = "true" ]; then
    SECONDS=0
    if stage_weights; then
        echo "staging: done in ${SECONDS}s"
        if [ -f "$JOB_SCRATCH/stage/.hf_ok" ]; then
            HF_HOME_SRC=$(jq -r '.defaults.staging.hf_home_src // empty' "$JSON_FILE")
            # HF_TOKEN_PATH defaults to HF_HOME/token; keep the source token reachable.
            if [ -z "${HF_TOKEN_PATH:-}" ] && [ -n "$HF_HOME_SRC" ] && [ -f "$HF_HOME_SRC/token" ]; then
                export HF_TOKEN_PATH="$HF_HOME_SRC/token"
            fi
            export HF_HOME="$JOB_SCRATCH/stage/huggingface"
            unset HF_HUB_CACHE HUGGINGFACE_HUB_CACHE
        fi
        if [ -f "$JOB_SCRATCH/stage/.tabpfn_ok" ]; then
            export TABPFN_MODEL_CACHE_DIR="$JOB_SCRATCH/stage/tabpfn"
        fi
    else
        echo "staging: weights served from the shared filesystem"
    fi
fi

# Optional page-cache pre-touch of the job's libraries, once per node boot (marker in the JIT root).
if [ "$PRETOUCH_LIBS" = "true" ]; then
    "$PYTHON_PATH" -P -m tabflow_slurm.node_prep pretouch \
        --packages "$(jq -r '.defaults.staging.pretouch_packages // [] | join(",")' "$JSON_FILE")" \
        --paths "$(jq -r '.defaults.staging.pretouch_paths // [] | join(",")' "$JSON_FILE")" \
        --max-bytes "$PRETOUCH_MAX_BYTES" \
        --marker "$JIT_CACHE_ROOT/pretouch.done" || true
fi

# Effective environment, through an allow-list (tokens and other secrets never reach the log).
env | grep -E '^(OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|NUMEXPR_MAX_THREADS|OMP_PROC_BIND|OMP_PLACES|CUDA_VISIBLE_DEVICES|PYTORCH_CUDA_ALLOC_CONF|HF_HOME|HF_HUB_CACHE|HF_TOKEN_PATH|HF_HUB_OFFLINE|HF_HUB_DISABLE_PROGRESS_BARS|TMPDIR|TABARENA_[A-Z_]*|TABPFN_MODEL_CACHE_DIR|TABPFN_DISABLE_TELEMETRY|NUMBA_CACHE_DIR|TRITON_CACHE_DIR|CUDA_CACHE_PATH|LD_LIBRARY_PATH|SLURM_JOB_ID|SLURM_ARRAY_JOB_ID|SLURM_ARRAY_TASK_ID|SLURM_CPUS_ON_NODE|SLURM_JOB_CPUS_PER_NODE|SLURM_MEM_PER_NODE|SLURM_GPUS)=' \
    | grep -vE 'TOKEN=|SECRET|PASSWORD|_KEY=' | sort || true

run_one() {
    local EXPERIMENT="$1"
    local DATASET="$2"
    local FOLD="$3"
    local REPEAT="$4"
    echo "Running experiment=$EXPERIMENT dataset=$DATASET fold=$FOLD repeat=$REPEAT"

    # -P keeps the script directory and cwd off sys.path (the venv install is the only tabarena).
    "$PYTHON_PATH" -P "$RUNSCRIPT" \
        --experiment "$EXPERIMENT" \
        --dataset "$DATASET" \
        --fold "$FOLD" \
        --repeat "$REPEAT" \
        --job_batch_dir "$JOB_BATCH_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --num_cpus "$NUM_CPUS" \
        --num_gpus "$NUM_GPUS" \
        --memory_limit "$MEMORY_LIMIT" \
        --setup_ray_for_slurm_shared_resources_environment "$SETUP_RAY" \
        --ignore_cache "$IGNORE_CACHE" \
        --ray_temp_root "$JOB_SCRATCH/ray" \
        --offline_weights "$OFFLINE_WEIGHTS" \
        --require_warmup "$REQUIRE_WARMUP"
}

# Bundle format: each job has `items: [...]` with one entry per (experiment, dataset, fold, repeat)
# work unit, streamed as TSV. A failing item is counted and the loop continues (the `if` disables
# errexit for run_one); three consecutive failures stop the bundle (a wedged GPU or a full disk
# should not burn every item). The task exits non-zero when any item failed.
NUM_ITEMS=$(jq -r --argjson J "$J" '.jobs[$J].items | length' "$JSON_FILE")
echo "Bundle items: $NUM_ITEMS"
MAX_CONSECUTIVE_FAILURES=3

FAILED=0
N_OK=0
N_FAILED=0
CONSECUTIVE=0
while IFS=$'\t' read -r EXPERIMENT DATASET FOLD REPEAT; do
    if run_one "$EXPERIMENT" "$DATASET" "$FOLD" "$REPEAT"; then
        N_OK=$((N_OK + 1))
        CONSECUTIVE=0
        echo "##### item OK: experiment=$EXPERIMENT dataset=$DATASET fold=$FOLD repeat=$REPEAT"
    else
        RC=$?
        FAILED=1
        N_FAILED=$((N_FAILED + 1))
        CONSECUTIVE=$((CONSECUTIVE + 1))
        echo "##### item FAILED (exit $RC): experiment=$EXPERIMENT dataset=$DATASET fold=$FOLD repeat=$REPEAT"
        if [ "$CONSECUTIVE" -ge "$MAX_CONSECUTIVE_FAILURES" ]; then
            echo "##### stopping bundle: $CONSECUTIVE consecutive failures"
            break
        fi
    fi
done < <(jq -r --argjson J "$J" \
    '.jobs[$J].items[] | [.experiment, .dataset, .fold, .repeat] | @tsv' \
    "$JSON_FILE")

echo "##### bundle summary: ok=$N_OK failed=$N_FAILED total=$NUM_ITEMS"
exit $FAILED
