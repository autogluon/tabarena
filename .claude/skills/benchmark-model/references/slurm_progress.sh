#!/usr/bin/env bash
# Progress of one SLURM array job launched by a `run_<model>.py setup` command.
#
# Prints one status line per interval and exits when the array has no running or queued task
# left. Meant to be driven by the Monitor tool (each line becomes a notification), or run once
# with --once for an on-demand snapshot.
#
#   slurm_progress.sh JOB_ID TOTAL_TASKS [--results-dir DIR] [--expected-results N]
#                     [--log-dir DIR] [--interval SECONDS] [--once]
#
#   JOB_ID              the id printed by sbatch ("Submitted batch job 1127523")
#   TOTAL_TASKS         array length, K+1 for `sbatch --array=0-K%N ...`
#   --results-dir       the run's <workspace>/output/<benchmark_name>/data; counts results.pkl
#   --expected-results  results.pkl files expected when everything is done (pre-existing cache
#                       hits + the "Approved N items" line printed by setup)
#   --log-dir           <workspace>/slurm_out/<benchmark_name>/<JOB_ID>, named in failure lines
#   --interval          seconds between status lines (default 600)
#   --once              print one status line and exit
#
# Exit code: 0 when every task completed, 2 when at least one task ended in a failure state.
# Terminal failure states: FAILED, TIMEOUT, OUT_OF_MEMORY, CANCELLED, BOOT_FAIL, DEADLINE.
# NODE_FAIL, PREEMPTED and REQUEUED are transient (the submit script sets --requeue) and are
# reported as "requeued", not failed.
set -u

job_id=${1:?usage: slurm_progress.sh JOB_ID TOTAL_TASKS [options]}
total=${2:?usage: slurm_progress.sh JOB_ID TOTAL_TASKS [options]}
shift 2
results_dir=""
expected_results=""
log_dir=""
interval=600
once=0
while [ $# -gt 0 ]; do
    case "$1" in
        --results-dir) results_dir=$2; shift 2 ;;
        --expected-results) expected_results=$2; shift 2 ;;
        --log-dir) log_dir=$2; shift 2 ;;
        --interval) interval=$2; shift 2 ;;
        --once) once=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 64 ;;
    esac
done

reported_failed=""

snapshot() {
    # One row per array task (allocations only; a requeued task shows its latest attempt).
    local rows
    rows=$(sacct -j "$job_id" -X -n -P --format=JobID,State,ExitCode 2>/dev/null || true)
    local done_n failed_n running_n requeued_n
    done_n=$(printf '%s\n' "$rows" | awk -F'|' '$2=="COMPLETED"' | wc -l)
    running_n=$(printf '%s\n' "$rows" | awk -F'|' '$2=="RUNNING" || $2=="COMPLETING"' | wc -l)
    requeued_n=$(printf '%s\n' "$rows" | awk -F'|' '$2=="NODE_FAIL" || $2=="PREEMPTED" || $2=="REQUEUED" || $2=="RESIZING"' | wc -l)
    local failed_rows
    failed_rows=$(printf '%s\n' "$rows" | awk -F'|' '$2 ~ /^(FAILED|TIMEOUT|OUT_OF_MEMORY|CANCELLED|BOOT_FAIL|DEADLINE)/')
    failed_n=$(printf '%s\n' "$failed_rows" | sed '/^$/d' | wc -l)
    # squeue -r lists queued array tasks one per line (sacct folds the not-yet-started ones).
    local queued_n
    queued_n=$(squeue -j "$job_id" -h -r -t PD 2>/dev/null | wc -l)
    local left pct
    left=$(( total - done_n - failed_n ))
    [ "$left" -lt 0 ] && left=0
    pct=$(awk -v l="$left" -v t="$total" 'BEGIN { if (t > 0) printf "%.1f", 100 * l / t; else print "0.0" }')

    local line
    line="[$(date +%H:%M)] job $job_id: ${pct}% of tasks left | done $done_n/$total, failed $failed_n, running $running_n, queued $queued_n, requeued $requeued_n"
    if [ -n "$results_dir" ] && [ -d "$results_dir" ]; then
        local n_results
        n_results=$(find "$results_dir" -name results.pkl 2>/dev/null | wc -l)
        if [ -n "$expected_results" ]; then
            line="$line | results.pkl $n_results/$expected_results"
        else
            line="$line | results.pkl $n_results"
        fi
    fi
    echo "$line"

    # Report each failed task once, with its state, exit code and log file.
    if [ "$failed_n" -gt 0 ]; then
        local new_failed=""
        while IFS='|' read -r jid state code; do
            [ -z "$jid" ] && continue
            case " $reported_failed " in *" $jid "*) continue ;; esac
            reported_failed="$reported_failed $jid"
            local task=${jid#*_}
            if [ -n "$log_dir" ]; then
                new_failed="$new_failed\n  $jid $state exit=$code log=$log_dir/slurm-${job_id}_${task}.out"
            else
                new_failed="$new_failed\n  $jid $state exit=$code"
            fi
        done <<< "$failed_rows"
        if [ -n "$new_failed" ]; then
            printf 'FAILED tasks (new):%b\n' "$new_failed"
        fi
    fi

    # Terminal when squeue lists nothing for the job (no running, completing or queued task).
    local active
    active=$(squeue -j "$job_id" -h -r 2>/dev/null | wc -l)
    if [ "$active" -eq 0 ]; then
        if [ "$failed_n" -gt 0 ]; then
            echo "DONE job $job_id with failures: completed $done_n/$total, failed $failed_n. Rerun '<script> setup' to re-approve only the missing items after fixing the cause."
            return 2
        fi
        echo "DONE job $job_id: completed $done_n/$total, no failures."
        return 0
    fi
    return 1
}

if [ "$once" -eq 1 ]; then
    snapshot
    rc=$?
    [ "$rc" -eq 1 ] && exit 0
    exit "$rc"
fi

while true; do
    snapshot
    rc=$?
    if [ "$rc" -ne 1 ]; then
        exit "$rc"
    fi
    sleep "$interval"
done
