#!/usr/bin/env bash
# Independent native Bandit pilot. Never loads a Strategic Blocking adapter.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PROJECT="${PROJECT:-$(cd -- "$script_dir/.." && pwd)}"
export RAGEN_DIR="${RAGEN_DIR:-$HOME/RAGEN}"
export PYTHONPATH="$PROJECT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mode="${1:-run}"
case "$mode" in
    run|--check-only) ;;
    *) echo 'Usage: bash scripts/run_bandit_bonus.sh [run|--check-only]' >&2; exit 2 ;;
esac
if [[ ! -f "$PROJECT/bandit_bonus.yaml" || ! -f "$RAGEN_DIR/train.py" ]]; then
    echo 'PROJECT or RAGEN_DIR is not the expected checkout.' >&2
    exit 1
fi
mkdir -p "$PROJECT/results/bandit"
run_dir=$(mktemp -d "$PROJECT/results/bandit/train-bandit-XXXXXXXX")
phase=preflight
on_failure() {
    code=$?
    trap - ERR
    python - "$run_dir" "$phase" "$code" <<'PY'
import json, sys
from pathlib import Path
Path(sys.argv[1], 'FAILED.json').write_text(json.dumps(
    {'phase': sys.argv[2], 'exit_code': int(sys.argv[3]), 'status': 'Incomplete; inspect stage logs.'}, indent=2) + '\n')
PY
    printf 'Stopped at %s. Keep the evidence directory: %s\n' "$phase" "$run_dir" >&2
    exit "$code"
}
trap on_failure ERR
printf 'Bandit output: %s\n' "$run_dir"
cd "$RAGEN_DIR"
python "$PROJECT/scripts/patch_ragen.py" --ragen-dir "$RAGEN_DIR" 2>&1 | tee "$run_dir/compatibility.log"
python "$PROJECT/scripts/verify_strategic_environment.py" --ragen-dir "$RAGEN_DIR" --output "$run_dir/environment-contribution.json"
python "$PROJECT/scripts/bandit_bonus.py" preflight --run "$run_dir" --ragen-dir "$RAGEN_DIR" 2>&1 | tee "$run_dir/preflight.log"
python -m pip freeze > "$run_dir/dependencies.txt"
git rev-parse HEAD > "$run_dir/ragen-revision.txt"
git diff --no-ext-diff > "$run_dir/ragen-working-tree.patch"
config_file=$(mktemp config/bandit_bonus_XXXXXXXX.yaml)
python - "$PROJECT" "$run_dir" "$config_file" <<'PY'
import json, sys, hashlib
from pathlib import Path
import yaml
project, run, config = map(Path, sys.argv[1:])
cfg = yaml.safe_load((project / 'bandit_bonus.yaml').read_text())
cfg['trainer']['default_local_dir'] = str(run / 'checkpoints')
cfg['trainer']['validation_data_dir'] = str(run / 'validation')
text = yaml.safe_dump(cfg, sort_keys=False)
config.write_text(text)
(run / 'config.yaml').write_text(text)
names = ['bandit_bonus.yaml', 'scripts/bandit_bonus.py', 'scripts/run_bandit_bonus.sh',
         'scripts/evaluate_checkpoint.py', 'scripts/export_pilot_adapter.py',
         'scripts/summarize_training.py']
(run / 'source-hashes.json').write_text(json.dumps({
    name: hashlib.sha256((project / name).read_bytes()).hexdigest() for name in names}, indent=2) + '\n')
PY
name=$(basename "$config_file" .yaml)
python train.py --config-name "$name" --cfg job > "$run_dir/composed.yaml"
python "$PROJECT/scripts/bandit_bonus.py" prepare --run "$run_dir" --ragen-dir "$RAGEN_DIR"
if [[ "$mode" == --check-only ]]; then
    printf 'CHECKS PASSED: native environments and composed configuration; no GPU training or model evaluation ran.\nEvidence: %s\n' "$run_dir"
    exit 0
fi
nvidia-smi > "$run_dir/nvidia-smi.txt"
phase=base-evaluation
python -u "$PROJECT/scripts/bandit_bonus.py" worker --policy base --run "$run_dir" --ragen-dir "$RAGEN_DIR" 2>&1 | tee "$run_dir/base.log"
phase=training
python -u train.py --config-name "$name" 2>&1 | tee "$run_dir/run.log"
phase=checkpoint-export
python -u "$PROJECT/scripts/bandit_bonus.py" export --run "$run_dir" --ragen-dir "$RAGEN_DIR" 2>&1 | tee "$run_dir/export.log"
phase=adapter-evaluation
python -u "$PROJECT/scripts/bandit_bonus.py" worker --policy adapter --run "$run_dir" --ragen-dir "$RAGEN_DIR" 2>&1 | tee "$run_dir/adapter.log"
phase=report
python "$PROJECT/scripts/bandit_bonus.py" report --run "$run_dir" --ragen-dir "$RAGEN_DIR" 2>&1 | tee "$run_dir/report.log"
printf 'Bandit pilot completed: %s/comparison.md\n' "$run_dir"
