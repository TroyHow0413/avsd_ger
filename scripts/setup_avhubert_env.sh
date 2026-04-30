#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AVHUBERT_DIR="${AVHUBERT_DIR:-${REPO_ROOT}/av_hubert}"
PRETRAINING_FILE="${AVHUBERT_DIR}/avhubert/hubert_pretraining.py"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: activate the avsdger conda env first, then rerun this script." >&2
  echo "       conda activate avsdger" >&2
  exit 1
fi

if [[ ! -f "${PRETRAINING_FILE}" ]]; then
  echo "ERROR: cannot find ${PRETRAINING_FILE}" >&2
  echo "       Clone AV-HuBERT into ${AVHUBERT_DIR}, or set AVHUBERT_DIR=/path/to/av_hubert." >&2
  exit 1
fi

if ! grep -q "input_modality:" "${PRETRAINING_FILE}"; then
  python - "${PRETRAINING_FILE}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
needle = '    fine_tuning: bool = field(default=False, metadata={"help": "set to true if fine-tuning AV-Hubert"})\n'
insert = needle + '    input_modality: Optional[str] = field(default="audiovisual", metadata={"help": "input modality: audio | video | audiovisual"})\n'
if needle not in text:
    raise SystemExit(f"Could not find insertion point in {path}")
path.write_text(text.replace(needle, insert, 1))
print(f"Patched {path}: added AVHubertPretrainingConfig.input_modality")
PY
else
  echo "AV-HuBERT input_modality compatibility field already present."
fi

HOOK_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
HOOK_FILE="${HOOK_DIR}/avhubert_path.sh"
mkdir -p "${HOOK_DIR}"
cat > "${HOOK_FILE}" <<EOF
export PYTHONPATH="${AVHUBERT_DIR}:${AVHUBERT_DIR}/avhubert:\${PYTHONPATH:-}"
EOF
echo "Wrote conda activate hook: ${HOOK_FILE}"

export PYTHONPATH="${AVHUBERT_DIR}:${AVHUBERT_DIR}/avhubert:${PYTHONPATH:-}"
python - <<'PY'
import avhubert.hubert_pretraining as hp
print("AV-HuBERT import:", hp.__file__)
print("input_modality field:", "input_modality" in hp.AVHubertPretrainingConfig.__dataclass_fields__)
PY

echo "Done. Restart the shell or run: conda deactivate && conda activate avsdger"
