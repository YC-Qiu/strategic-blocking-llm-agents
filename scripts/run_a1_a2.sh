#!/usr/bin/env bash
# Run from the existing Ubuntu RAGEN environment. No commits or pushes are made.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
handoff_dir=$(cd -- "$script_dir/.." && pwd)
export PROJECT="${PROJECT:-$handoff_dir}"
export RAGEN_DIR="${RAGEN_DIR:-$HOME/RAGEN}"
mode="${1:-all}"
case "$mode" in
    all|a1|a2) ;;
    *) echo "Usage: bash scripts/run_a1_a2.sh [all|a1|a2]" >&2; exit 2 ;;
esac
if [[ ! -f "$PROJECT/scripts/run_ragen.sh" || ! -f "$RAGEN_DIR/train.py" ]]; then
    echo "Set PROJECT and RAGEN_DIR to the existing project and RAGEN checkout." >&2
    exit 1
fi
python -c 'import torch, peft; print("Python dependencies available:", torch.__version__, peft.__version__)'
mkdir -p "$PROJECT/results"
evidence_dir=$(mktemp -d "$PROJECT/results/a1-a2-XXXXXXXX")
printf 'Evidence directory: %s\n' "$evidence_dir"
printf '%s\n' "$evidence_dir" > "$PROJECT/results/latest-a1-a2-evidence-path.txt"

if [[ "$mode" == all || "$mode" == a1 ]]; then
    nvidia-smi > "$evidence_dir/nvidia-smi.txt"
    df -h "$PROJECT" "$RAGEN_DIR" /tmp > "$evidence_dir/disk-space.txt"
    git -C "$PROJECT" rev-parse HEAD > "$evidence_dir/project-revision.txt"
    git -C "$PROJECT" status --short > "$evidence_dir/project-status.txt"
    git -C "$RAGEN_DIR" rev-parse HEAD > "$evidence_dir/ragen-revision.txt"
    git -C "$RAGEN_DIR" status --short > "$evidence_dir/ragen-status.txt"
    git -C "$RAGEN_DIR" diff --no-ext-diff --binary HEAD > "$evidence_dir/ragen-local-changes.patch"
    python -m pip freeze > "$evidence_dir/dependencies.txt"
    python -m pip check > "$evidence_dir/pip-check.txt" 2>&1 || true
    bash -n "$PROJECT/scripts/run_ragen.sh"
    (
        cd "$PROJECT"
        python -m unittest -v
    ) 2>&1 | tee "$evidence_dir/environment-tests.log"
    # The full launcher already includes preflight; do not run it twice.
    bash "$PROJECT/scripts/run_ragen.sh" train-one 2>&1 | tee "$evidence_dir/packaged-training.log"
    git -C "$RAGEN_DIR" diff --no-ext-diff --binary HEAD > "$evidence_dir/ragen-local-changes-after.patch"
    printf 'A1 launcher finished. Review the rollout/update/save log before marking A1 complete.\n'
fi

if [[ "$mode" == all || "$mode" == a2 ]]; then
    actor_dir="$PROJECT/results/train/train-one-F5JCCw3d/checkpoints/global_step_1/actor"
    python "$handoff_dir/scripts/export_pilot_adapter.py" \
        --actor-dir "$actor_dir" --output "$evidence_dir/recovered_adapter" \
        2>&1 | tee "$evidence_dir/adapter-export.log"
    python "$handoff_dir/scripts/verify_recovered_adapter.py" \
        --actor-dir "$actor_dir" --adapter-dir "$evidence_dir/recovered_adapter" \
        --project "$PROJECT" --report "$evidence_dir/adapter-verification.json" \
        2>&1 | tee "$evidence_dir/adapter-verification.log"
fi
printf 'Finished requested stages. Evidence: %s\n' "$evidence_dir"
