"""Frozen prospective evaluation maps; excluded from every training launcher mode."""
import hashlib
import json
from pathlib import Path

SUITE_PATH = Path(__file__).parent / "evaluation/frozen-transfer-v1.json"
SUITE_SHA256 = "13d836c10a14a7047c412e2afee4496a113314b940716a6de051a7092e3e66a8"


def load_suite():
    raw = SUITE_PATH.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SUITE_SHA256:
        raise ValueError("Frozen suite was modified; create a new version instead.")
    return json.loads(raw)


FROZEN_MAPS = {item["name"]: item["layout"] for item in load_suite()["maps"]}
