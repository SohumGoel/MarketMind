#!/usr/bin/env bash
# Evaluate all checkpoints in outputs/runs/ and write a combined results summary.
set -e

RUNS_DIR="outputs/runs"
SUMMARY_FILE="outputs/results_summary.json"

if [ ! -d "$RUNS_DIR" ]; then
    echo "No runs found at $RUNS_DIR. Train first with run_ablations.sh"
    exit 1
fi

echo "[" > "$SUMMARY_FILE"
FIRST=true

for RUN_DIR in "$RUNS_DIR"/*/; do
    # Find best checkpoint directory
    CHECKPOINT=$(ls -d "$RUN_DIR"checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -z "$CHECKPOINT" ]; then
        # Fall back to run dir itself if no checkpoint subdirs
        CHECKPOINT="$RUN_DIR"
    fi

    echo "Evaluating: $CHECKPOINT"
    python evaluation/evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --split test \
        --max_samples 200

    RESULT_FILE="$CHECKPOINT/results_finetuned.json"
    if [ -f "$RESULT_FILE" ]; then
        if [ "$FIRST" = false ]; then
            echo "," >> "$SUMMARY_FILE"
        fi
        cat "$RESULT_FILE" >> "$SUMMARY_FILE"
        FIRST=false
    fi
done

echo "]" >> "$SUMMARY_FILE"
echo "Results summary written to $SUMMARY_FILE"
