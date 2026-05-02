#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
AVHUBERT_DIR="${AVHUBERT_DIR:-${REPO_ROOT}/av_hubert}"
PRETRAINING_FILE="${AVHUBERT_DIR}/avhubert/hubert_pretraining.py"
HUBERT_FILE="${AVHUBERT_DIR}/avhubert/hubert.py"
HUBERT_ASR_FILE="${AVHUBERT_DIR}/avhubert/hubert_asr.py"

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

python - "${HUBERT_FILE}" "${HUBERT_ASR_FILE}" <<'PY'
from pathlib import Path
import sys

hubert_path = Path(sys.argv[1])
hubert_asr_path = Path(sys.argv[2])

replacements = {
    hubert_path: {
        "from hubert_pretraining import": "from .hubert_pretraining import",
        "from resnet import ResEncoder": "from .resnet import ResEncoder",
        "from utils import compute_mask_indices": "from .utils import compute_mask_indices",
        "from decoder import TransformerDecoder": "from .decoder import TransformerDecoder",
    },
    hubert_asr_path: {
        "from hubert import AVHubertModel": "from .hubert import AVHubertModel",
        "from decoder import TransformerDecoder": "from .decoder import TransformerDecoder",
    },
}

for path, reps in replacements.items():
    if not path.exists():
        raise SystemExit(f"Cannot find {path}")
    text = path.read_text()
    new = text
    for old, replacement in reps.items():
        new = new.replace(old, replacement)
    if new != text:
        path.write_text(new)
        print(f"Patched relative imports in {path}")
    else:
        print(f"Relative imports already OK in {path}")

text = hubert_path.read_text()
if "required_seq_len_multiple:" not in text:
    needle = "    no_scale_embedding: bool = field(default=True, metadata={'help': 'scale embedding'})\n"
    insert = needle + '''
    # Fields added for fairseq >= 0.12.2 compatibility (TransformerEncoder reads these)
    required_seq_len_multiple: int = field(
        default=2,
        metadata={"help": "pad encoder input so sequence length is divisible by this"},
    )
    layer_type: str = field(
        default="transformer",
        metadata={"help": "transformer | conformer"},
    )
    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )
    # Conformer-specific fields are only used when layer_type=conformer.
    depthwise_conv_kernel_size: int = field(default=31, metadata={"help": "conformer depthwise conv kernel size"})
    attn_type: str = field(default="", metadata={"help": "attention type; empty = standard MHA"})
    pos_enc_type: str = field(default="abs", metadata={"help": "positional encoding type for conformer"})
    fp16: bool = field(default=False, metadata={"help": "fp16 flag passed to conformer layer"})
'''
    if needle not in text:
        raise SystemExit(f"Could not find AVHubertConfig insertion point in {hubert_path}")
    hubert_path.write_text(text.replace(needle, insert, 1))
    print(f"Patched fairseq TransformerEncoder compatibility fields in {hubert_path}")
else:
    print(f"Fairseq TransformerEncoder compatibility fields already present in {hubert_path}")

text = hubert_asr_path.read_text()
if "encoder output dimension for decoder cross-attention" not in text:
    needle = "class AVHubertSeq2SeqConfig(AVHubertAsrConfig):\n"
    insert = needle + '''    encoder_embed_dim: int = field(
        default=1024,
        metadata={"help": "encoder output dimension for decoder cross-attention"},
    )
'''
    if needle not in text:
        raise SystemExit(f"Could not find AVHubertSeq2SeqConfig insertion point in {hubert_asr_path}")
    text = text.replace(needle, insert, 1)
    hubert_asr_path.write_text(text)
    print(f"Patched AVHubertSeq2SeqConfig.encoder_embed_dim in {hubert_asr_path}")
else:
    print(f"AVHubertSeq2SeqConfig.encoder_embed_dim already present in {hubert_asr_path}")

text = hubert_asr_path.read_text()
if "Large VSR checkpoints use a 1024-d encoder" not in text:
    needle = """        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)
"""
    insert = """        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx=padding_idx)
            return emb

        # Fairseq's TransformerDecoder defaults encoder_embed_dim to 512 when
        # the field is absent from older AV-HuBERT seq2seq checkpoints/configs.
        # Large VSR checkpoints use a 1024-d encoder, so leaving the default
        # creates 512-d cross-attention projections and checkpoint loading fails.
        if getattr(cfg, "encoder_embed_dim", None) in (None, 512):
            with open_dict(cfg):
                cfg.encoder_embed_dim = getattr(
                    getattr(encoder.w2v_model, "cfg", None),
                    "encoder_embed_dim",
                    cfg.decoder_embed_dim,
                )

        decoder_embed_tokens = build_embedding(tgt_dict, cfg.decoder_embed_dim)
"""
    if needle not in text:
        raise SystemExit(f"Could not find decoder insertion point in {hubert_asr_path}")
    hubert_asr_path.write_text(text.replace(needle, insert, 1))
    print(f"Patched seq2seq decoder encoder_embed_dim compatibility in {hubert_asr_path}")
else:
    print(f"Seq2seq decoder encoder_embed_dim compatibility already present in {hubert_asr_path}")
PY

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
