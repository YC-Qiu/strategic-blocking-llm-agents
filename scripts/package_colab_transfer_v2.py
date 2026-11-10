"""Package reviewed code and two verified adapters for a private Colab handoff."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from frozen_eval import load_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    suite = load_suite()
    names = ["maze.py", "demo.py", "curriculum.py", "ragen_maze.py", "frozen_eval.py",
             "scripts/evaluate_colab_transfer.py", "scripts/evaluate_checkpoint.py", "scripts/validate_frozen_suite.py",
             "evaluation/frozen-transfer-v1.json", "evaluation/colab-protocol-v1.json", "docs/colab-transfer.md", "colab_protocol.py", "scripts/evaluate_colab_transfer_v2.py",
             "evaluation/colab-protocol-v2.json", "evaluation/colab-ragen-contract-v2.json",
             "docs/colab-transfer-v2.md", "docs/colab-v1-format-audit.json",
             "notebooks/colab-transfer-v2.ipynb", "test_colab_v2.py"]
    files = {name: ROOT / name for name in names}
    for policy, run in [("before", "paired-eval-fsx8vobm"), ("after", "paired-eval-trum2mve")]:
        for name, expected in suite["policies"][policy]["files"].items():
            source = args.results_root / run / "verified_adapter" / name
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Adapter does not match frozen evaluation: {source}")
            files[f"adapters/{policy}/{name}"] = source
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in files.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, source in files.items():
            archive.write(source, name)
        archive.writestr("bundle-manifest.json", json.dumps(hashes, indent=2) + "\n")
    print(f"Created {args.output}; {args.output.stat().st_size / 1024**2:.1f} MiB")
    print("Share privately with your teammate. This contains adapters, not full checkpoints.")


if __name__ == "__main__":
    main()
