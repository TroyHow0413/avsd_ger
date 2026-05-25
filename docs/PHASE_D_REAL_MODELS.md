# Deprecated: Phase D-G Real-Model Notes

This file is kept only so old links do not break.

The current real-model workflow is documented in [`REAL_MODEL_WORKFLOW.md`](REAL_MODEL_WORKFLOW.md). The old Phase 0/A/B/C/D/E/F/G rollout notes were moved to [`LEGACY_PHASE_ROLLOUT.md`](LEGACY_PHASE_ROLLOUT.md).

For current training, use:

```bash
python scripts/train_identity.py --config configs/default.yaml --manifest data/your_real_train_manifest.jsonl --out checkpoints/stage1/
python scripts/train_stage2.py --config configs/default.yaml --manifest data/your_real_train_manifest.jsonl --stage1-pool checkpoints/stage1/identity_pool_stage1.pt --out checkpoints/stage2/ --warmup joint
```
