"""Load a configured GER model and optionally run one teacher-forced backward step."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from avsd_ger.c2_alignment import GERHead
from avsd_ger.training.ger_loss import GERCrossEntropy
from avsd_ger.utils import load_config, resolve_device, seed_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_all(int(cfg.get("seed", 1337)))
    device = resolve_device(cfg.get("device", "cuda"))
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    ger = GERHead(
        cfg["ger"],
        z_dim=int(cfg["identity"]["fused_dim"]),
        d_align=int(cfg["alignment"]["d_model"]),
        stub=False,
        device=device,
    )
    trainable = sum(p.numel() for p in ger.parameters() if p.requires_grad)
    total = sum(p.numel() for p in ger.parameters())
    result: dict[str, object] = {
        "model_family": ger.backend.profile.family,
        "model_path": ger.backend.model_path,
        "hidden_size": ger._llm_embed_dim,
        "dtype": str(next(ger._llm.parameters()).dtype),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "backward_requested": bool(args.backward),
    }

    if args.backward:
        ger.train()
        z_id = torch.randn(int(cfg["identity"]["fused_dim"]), device=device)
        f_align = torch.randn(4, int(cfg["alignment"]["d_model"]), device=device)
        report = GERCrossEntropy(ger)(
            z_id=z_id,
            f_align=f_align,
            nbest=["hello from the meeting", "hello meeting"],
            lip_hyp="hello from meeting",
            target="hello from the meeting",
            speaker_id="preflight_speaker",
            mode="av",
            use_av_context=True,
        )
        report.loss.backward()
        gradients = [
            parameter.grad
            for parameter in ger.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        result.update({
            "loss": float(report.loss.detach()),
            "target_tokens": int(report.n_target_tokens),
            "gradient_tensors": len(gradients),
            "all_gradients_finite": bool(
                gradients and all(torch.isfinite(gradient).all().item() for gradient in gradients)
            ),
            "nonzero_gradient_tensors": sum(
                int(torch.count_nonzero(gradient).item() > 0) for gradient in gradients
            ),
        })

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        result.update({
            "gpu": torch.cuda.get_device_name(device),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        })
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
