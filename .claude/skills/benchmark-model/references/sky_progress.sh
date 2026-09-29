#!/usr/bin/env bash
# Progress of one SkyPilot launch made by a `run_<model>.py setup --scheduler skypilot[-pool]` command.
#
# Prints one status line per interval from the bucket queue's markers and exits when every bundle
# is done, or when no worker job of the launch is left. Meant to be driven by the Monitor tool (each
# line becomes a notification), or run once with --once for an on-demand snapshot.
#
#   sky_progress.sh QUEUE_URI N_BUNDLES [--launch LAUNCH_ID] [--pool POOL]... [--sky SKY_BINARY]
#                   [--interval SECONDS] [--once]
#
#   QUEUE_URI     gs://.../runs/<benchmark>/queue/<launch_id>  (first line of the printed command block)
#   N_BUNDLES     the bundle count from the same line
#   --launch      the launch id; adds the `sky jobs queue` states of its worker jobs to each line
#   --pool        a pool the launch runs on (repeatable); adds its READY worker count to each line and
#                 flags, once per worker, a worker cluster name the pool controller finds on more than one
#                 VM (two workers collide when the pool name is too long). The controller's prober then
#                 fails on every pass and stops promoting finished workers to READY, so the pool idles
#                 below its size while every VM bills (see the skill's Step 6).
#   --sky         the sky executable (default: sky on PATH)
#   --interval    seconds between status lines (default 600)
#   --once        print one status line and exit
#
# Exit code: 0 when every bundle is done and no item failed, 2 when at least one item failed,
# 3 when the workers are gone but bundles remain (orphaned claims: rerun `setup` to re-enumerate).
set -u

queue=${1:?usage: sky_progress.sh QUEUE_URI N_BUNDLES [options]}
total=${2:?usage: sky_progress.sh QUEUE_URI N_BUNDLES [options]}
shift 2
launch=""
pools=""
sky_bin="sky"
interval=600
once=0
while [ $# -gt 0 ]; do
    case "$1" in
        --launch) launch=$2; shift 2 ;;
        --pool) pools="$pools $2"; shift 2 ;;
        --sky) sky_bin=$2; shift 2 ;;
        --interval) interval=$2; shift 2 ;;
        --once) once=1; shift ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done
queue=${queue%/}

count_markers() {
    # $1: the marker prefix; prints the number of bundle-level markers (no dot) and item-level markers (with a dot).
    local listing bundles items
    listing=$(gcloud storage ls "$queue/$1/" 2>/dev/null | sed "s|^$queue/$1/||")
    bundles=$(printf '%s\n' "$listing" | grep -c -v -E '\.|^$')
    items=$(printf '%s\n' "$listing" | grep -c '\.')
    echo "$bundles $items"
}

seen_failed=""
seen_duplicates=""
while true; do
    read -r done_bundles done_items < <(count_markers done)
    read -r _ failed_items < <(count_markers failed)
    claimed=$(gcloud storage ls "$queue/claims/" 2>/dev/null | grep -c . || true)
    left=$(( total - done_bundles ))
    pct=$(( total > 0 ? 100 * left / total : 0 ))
    states=""
    running=""
    if [ -n "$launch" ]; then
        # One "STATUS" word per worker job of this launch, summarized as counts.
        states=$("$sky_bin" jobs queue 2>/dev/null | awk -v name="$launch" '$0 ~ name {print $0}' \
            | grep -o -E 'PENDING|STARTING|RUNNING|RECOVERING|SUCCEEDED|FAILED[A-Z_]*|CANCELLED|CANCELLING' \
            | sort | uniq -c | awk '{printf "%s=%s ", $2, $1}')
        running=$(printf '%s' "$states" | grep -c -E 'PENDING|STARTING|RUNNING|RECOVERING' || true)
    fi
    ready=""
    for pool in $pools; do
        ready="$ready $pool=$("$sky_bin" jobs pool status "$pool" 2>/dev/null | awk -v p="$pool" '$1 == p' \
            | grep -o -E '[0-9]+/[0-9]+' | tail -1)"
    done
    echo "$(date '+%H:%M:%S') ${pct}% of bundles left | bundles done=${done_bundles}/${total} claimed=${claimed} | items done=${done_items} failed=${failed_items} ${states:+| jobs: $states}${ready:+| ready:$ready}"
    # A worker provisioned twice: the controller logs its cluster name on every probe.
    for pool in $pools; do
        for cluster in $("$sky_bin" jobs pool logs --controller --no-follow --tail 300 "$pool" 2>/dev/null \
            | grep -o -E "same cluster name tag in the cloud provider for cluster '[^']+'" \
            | grep -o -E "'[^']+'" | tr -d "'" | sort -u); do
            case " $seen_duplicates " in *" $cluster "*) continue ;; esac
            seen_duplicates="$seen_duplicates $cluster"
            echo "  DUPLICATE VM for pool worker $cluster: two workers share one GCP cluster name (a pool name over" \
                "22 characters); cancel the launch, take the pool down, rerun setup with a shorter pool_name" \
                "(benchmark-model skill, Step 6)"
        done
    done
    # Report each failed item once, with its recorded status and coordinates.
    for marker in $(gcloud storage ls "$queue/failed/" 2>/dev/null); do
        case " $seen_failed " in *" $marker "*) continue ;; esac
        seen_failed="$seen_failed $marker"
        echo "  FAILED $(basename "$marker"): $(gcloud storage cat "$marker" 2>/dev/null | head -1)"
    done
    if [ "$done_bundles" -ge "$total" ]; then
        if [ "$failed_items" -gt 0 ]; then echo "DONE with $failed_items failed item(s)"; exit 2; fi
        echo "DONE all $total bundle(s)"; exit 0
    fi
    if [ -n "$launch" ] && [ "$running" = "0" ] && [ -n "$states" ]; then
        echo "WORKERS GONE with $left bundle(s) left (orphaned claims: rerun setup to re-enumerate)"; exit 3
    fi
    [ "$once" = "1" ] && exit 0
    sleep "$interval"
done
