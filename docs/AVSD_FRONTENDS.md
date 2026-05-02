# AVSD Frontend Profiles

This repo's core model is **frontend-agnostic**: it consumes turn-level
manifests and focuses on identity-conditioned alignment, GER correction, and
confidence-gated feedback. A raw multi-speaker meeting video still needs an
upstream AVSD frontend that proposes turns, active speaker tracks, and mouth
ROIs.

> Paper note: do not claim that the current pipeline solves raw-video
> diarization end to end. Claim that AVSD-GER improves speaker-aware
> recognition under multiple frontend quality levels.

## Recommended Claim

Use this framing:

> Our framework is not designed to compete with diarization frontends. Instead,
> it is frontend-agnostic: given diarization hypotheses of varying quality, it
> improves speaker-aware recognition through identity-conditioned alignment,
> generative correction, and confidence-gated feedback.

In Chinese:

> 我们不把贡献点放在切分最强，而是证明即使前端切分不是最强，后端的身份条件对齐和闭环纠错仍然能提高最终 speaker-aware transcript 的正确率。

## Frontend Conditions

| Profile | Purpose | Diarization | Active speaker | What it proves |
|---|---|---|---|---|
| `oracle_turns` | Upper bound | Ground-truth segment boundaries | Ground-truth or verified face tracks | C1/C2/C3 gains when segmentation is not the bottleneck |
| `common_pyannote_lightasd` | Main practical baseline | `pyannote/speaker-diarization-community-1` | Light-ASD or TalkNet | The framework works with a common open-source frontend |
| `strong_sortformer_talknet` | Strong/SOTA-ish reference | NVIDIA Sortformer v2.1 or pyannote Precision-2 | TalkNet or Light-ASD | The framework still helps when the frontend is already strong |
| `degraded_pyannote` | Robustness side proof | pyannote community + synthetic boundary/speaker noise | Light-ASD/TalkNet with optional track noise | The framework does not require perfect splitting ability |

The same list is available from code:

```bash
python scripts/frontend_profiles.py --format markdown
python scripts/frontend_profiles.py --format json
```

`--format markdown` is for reading or pasting into notes/papers. `--format json`
is for scripts, dashboards, or experiment metadata.

These commands only print the frontend registry. They do **not** start an
experiment. The experiment entry point remains `scripts/eval_ablations.py`.

PowerShell examples:

```powershell
# Single manifest
python scripts\eval_ablations.py `
  --config configs\default.yaml `
  --manifest data\ami_test\manifests\ES2004a.json `
  --pool checkpoints\identity_pool.pt `
  --out out\ami_ablation_ES2004a.json `
  --frontend-profile common_pyannote_lightasd `
  --no-power

# Whole manifest directory; writes one JSON per meeting plus summary.json
python scripts\eval_ablations.py `
  --config configs\default.yaml `
  --manifest data\ami_test\manifests `
  --pool checkpoints\identity_pool.pt `
  --out out\ami_ablation `
  --frontend-profile common_pyannote_lightasd `
  --no-power

# Glob pattern
python scripts\eval_ablations.py `
  --config configs\default.yaml `
  --manifest "data\ami_test\manifests\ES2004*.json" `
  --pool checkpoints\identity_pool.pt `
  --out out\ami_ablation_es2004 `
  --frontend-profile oracle_turns `
  --no-power
```

## Why These Choices

### Common Backbone

Use `pyannote/speaker-diarization-community-1` as the "everyone can reproduce
this" diarization backbone. It is open-source, local, widely used, and strong
enough that improvements on top of it are meaningful.

Pair it with:

- Light-ASD or TalkNet for active speaker detection.
- RetinaFace or InsightFace for face detection.
- SORT, ByteTrack, or DeepSORT for face-track association.
- AV-HuBERT `align_mouth.py` for mouth ROI extraction.

### Strong Frontend

Use either NVIDIA Sortformer v2.1 or pyannote Precision-2 as the strong
diarization reference. This row is not the core contribution; it is a control
showing that AVSD-GER does not only compensate for a weak frontend.

### Degraded Frontend

Start from the common frontend and perturb it:

- Add boundary jitter, for example +/- 0.25 s or +/- 0.50 s.
- Drop a small percentage of turns.
- Swap a percentage of speaker labels.
- Add noisy active-speaker track assignment.

This is the side proof for robustness: even when segmentation is imperfect,
C1/C2/C3 should reduce speaker confusion and improve transcript quality.

## Raw Video Contract

A raw-video frontend should eventually produce a manifest shaped like this:

```json
{
  "speakers": [
    {"speaker_id": "spk_01", "enrollment_audio": "data/enroll/spk_01.wav"}
  ],
  "turns": [
    {
      "turn_id": "m001_t0001",
      "start": 12.4,
      "end": 16.8,
      "audio": "data/turns/m001_t0001.wav",
      "mouth_roi": "data/turns/m001_t0001_mouth.npy",
      "speaker_mask_v": null,
      "ref_text": "only for evaluation",
      "ref_speaker": "only for evaluation"
    }
  ]
}
```

Required for inference:

- `turns[].audio`
- `turns[].mouth_roi`

Recommended for full AVSD experiments:

- `turns[].speaker_mask_v`
- `turns[].snr_per_tok`
- `turns[].lip_conf_v`
- frontend metadata, e.g. `frontend.profile`

## Suggested Experiment Table

| Row | Frontend | C1 | C2 | C3 | Main metric |
|---|---|---|---|---|---|
| Upper bound | `oracle_turns` | on | on | on | SA-WER / SCR |
| Common frontend | `common_pyannote_lightasd` | on | on | on | SA-WER / SCR |
| Strong frontend | `strong_sortformer_talknet` | on | on | on | SA-WER / SCR |
| Degraded frontend | `degraded_pyannote` | on | on | on | SA-WER / SCR |
| No identity | common frontend | off | on | on | AV-SID Acc / SCR |
| No GER | common frontend | on | off | on | WER / SA-WER |
| No feedback | common frontend | on | on | off | SA-WER / pool update safety |

Expected story:

1. Oracle turns show the upper-bound gain of the proposed C1/C2/C3 stack.
2. Common frontend shows the method is useful with a normal open-source
   diarization/ASD backbone.
3. Strong frontend shows the method is not merely patching bad segmentation.
4. Degraded frontend shows robustness when turn splitting is imperfect.

## Current Repo Status

Implemented now:

- Turn-level inference through `scripts/run_sample.py`.
- Session-level evaluation through `scripts/eval_ablations.py`.
- Frontend profile metadata in `avsd_ger/frontend/registry.py`.
- Frontend metadata emitted in ablation JSON output.

Not implemented yet:

- Raw full-video ingestion.
- Face detection/tracking.
- Active speaker detection.
- Automatic mouth ROI extraction for arbitrary meeting video.
- System-output RTTM diarization.
