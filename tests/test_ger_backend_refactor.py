import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from avsd_ger.c2_alignment import CheckpointCompatibilityError, GERHead
from avsd_ger.c2_alignment.generation_policy import GenerationPolicy
from avsd_ger.c2_alignment.model_backend import (
    LocalHFCausalLMBackend,
    get_model_profile,
)
from avsd_ger.utils import load_config


def ger_config(family: str = "qwen2.5-3b-instruct") -> dict:
    return {
        "backend": "fake",
        "model_family": family,
        "max_new_tokens": 8,
        "speaker_special_token": "[Speaker: ID_i]",
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": "auto"},
        "bridge": {"n_queries": 4, "n_heads": 2},
        "prompt_template": (
            "{speaker_tag}\nAudio hypothesis: {asr_nbest}\n"
            "Visual hypothesis: {lip_hyp}\n"
            "Aligned feature context: <AV_CTX>\nOutput:\n"
        ),
    }


class GERBackendRefactorTest(unittest.TestCase):
    def test_model_profiles_drive_hidden_size_and_lora_targets(self):
        qwen = GERHead(ger_config(), z_dim=8, d_align=16, stub=True)
        llama = GERHead(
            ger_config("llama-3.2-3b-instruct"),
            z_dim=8,
            d_align=16,
            stub=True,
        )

        self.assertEqual(qwen.id_proj.out_features, 2048)
        self.assertEqual(llama.id_proj.out_features, 3072)
        self.assertIn("gate_proj", get_model_profile(qwen.cfg).lora_target_modules)
        self.assertIn("down_proj", get_model_profile(llama.cfg).lora_target_modules)

    def test_prompt_uses_tokenizer_chat_template_and_preserves_placeholder(self):
        head = GERHead(ger_config(), z_dim=8, d_align=16, stub=True)

        prompt = head._render_text(
            "speaker-a", ["hello", "yellow"], "hello", mode="av"
        )

        self.assertTrue(prompt.startswith("<|user|>"))
        self.assertIn("hello | yellow", prompt)
        self.assertIn("<AV_CTX>", prompt)
        self.assertTrue(prompt.endswith("<|assistant|>\n"))

    def test_bridge_shape_tracks_selected_hidden_size(self):
        head = GERHead(ger_config(), z_dim=8, d_align=16, stub=True)
        prompt = head._render_text(None, ["hello"], "", mode="av")

        embeds = head._inputs_embeds(
            torch.zeros(8), torch.zeros(3, 16), prompt
        )

        self.assertEqual(embeds.ndim, 3)
        self.assertEqual(embeds.shape[0], 1)
        self.assertEqual(embeds.shape[-1], 2048)

    def test_fake_backend_keeps_legacy_stub_generation(self):
        head = GERHead(ger_config(), z_dim=8, d_align=16, stub=True)

        result = head.generate(
            torch.zeros(8), torch.zeros(3, 16), ["first", "second"]
        )

        self.assertEqual(result["text"], "first")
        self.assertEqual(result["raw_text"], "first")
        self.assertEqual(result["token_logprobs"].numel(), 0)

    def test_checkpoint_rejects_different_model_family(self):
        qwen = GERHead(ger_config(), z_dim=8, d_align=16, stub=True)
        llama = GERHead(
            ger_config("llama-3.2-3b-instruct"),
            z_dim=8,
            d_align=16,
            stub=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "projectors.pt"
            qwen.save_projector_checkpoint(path)

            with self.assertRaisesRegex(
                CheckpointCompatibilityError, "model_family"
            ):
                llama.load_projector_checkpoint(path)

    def test_local_backend_rejects_missing_path_without_downloading(self):
        cfg = ger_config()
        cfg.update({"backend": "local_hf", "model_path": "missing/model/path"})

        with self.assertRaisesRegex(FileNotFoundError, "never downloads"):
            LocalHFCausalLMBackend(cfg, torch.device("cpu"))

    def test_generation_policy_uses_only_scored_suffix(self):
        scores = (torch.tensor([[0.0, 2.0, 1.0]]), torch.tensor([[3.0, 0.0, 1.0]]))
        output = SimpleNamespace(
            sequences=torch.tensor([[99, 98, 1, 0]]), scores=scores
        )

        ids = GenerationPolicy.generated_ids(output)
        token_lp = GenerationPolicy.token_logprobs(output, ids)

        self.assertEqual(ids.tolist(), [1, 0])
        self.assertEqual(token_lp.shape, (2,))

    def test_public_configs_switch_supported_profiles(self):
        qwen = load_config("configs/qwen25_3b.yaml")["ger"]
        llama = load_config("configs/llama32_3b.yaml")["ger"]

        self.assertEqual(qwen["model_family"], "qwen2.5-3b-instruct")
        self.assertEqual(llama["model_family"], "llama-3.2-3b-instruct")
        self.assertEqual(qwen["lora"]["target_modules"], "auto")
        self.assertEqual(llama["lora"]["target_modules"], "auto")


if __name__ == "__main__":
    unittest.main()
