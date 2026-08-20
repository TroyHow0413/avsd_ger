import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from avsd_ger.c1_identity.identity_pool import IdentityPool
from avsd_ger.c2_alignment import CheckpointCompatibilityError, GERHead
from avsd_ger.c2_alignment.model_backend import (
    FakeCausalLM, FakeTokenizer, LocalHFCausalLMBackend, get_model_profile,
)
from avsd_ger.c3_feedback.ger_safety import GERSafetyGate
from avsd_ger.utils import load_config


def ger_cfg(family="qwen2.5-3b-instruct", *, heads=2):
    return {
        "backend": "fake", "model_family": family, "max_new_tokens": 8,
        "speaker_special_token": "[Speaker: ID_i]",
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0,
                 "bias": "none", "target_modules": "auto"},
        "bridge": {"n_queries": 4, "n_heads": heads},
        "prompt_template": "{speaker_tag} {asr_nbest} <AV_CTX> {lip_hyp}",
    }


class CapturingLM(FakeCausalLM):
    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(sequences=torch.tensor([[1]]), scores=())

    def forward(self, **kwargs):
        self.kwargs = kwargs
        batch, length = kwargs["inputs_embeds"].shape[:2]
        return SimpleNamespace(logits=torch.zeros(batch, length, 258,
                                                   device=kwargs["inputs_embeds"].device))


class CapturingBackend:
    kind = "local_hf"

    def __init__(self, cfg):
        self.profile = get_model_profile(cfg)
        self.tokenizer = FakeTokenizer(cfg["speaker_special_token"])
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = CapturingLM(self.profile.hidden_size)
        self.hidden_size = self.profile.hidden_size
        self.speaker_token_id = self.tokenizer._speaker_id
        self.lora_target_modules = self.profile.lora_target_modules


class IdentityFreshPoolTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {"top_k": 3, "min_av_consistency": 0.0,
                    "voice_dim": 3, "face_dim": 4, "fused_dim": 2}

    def test_fresh_load_retains_fuser_but_clears_cross_meeting_gallery(self):
        source = IdentityPool(self.cfg)
        source.enroll("train-speaker", torch.ones(3), torch.ones(4))
        with torch.no_grad():
            source.fuser.voice_proj.bias.fill_(0.42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.pt"
            source.save(path)
            fresh = IdentityPool(self.cfg)
            fresh.enroll("stale", torch.zeros(3), torch.zeros(4))
            fresh.load(path, load_gallery=False)
            self.assertEqual(fresh.speaker_ids, ())
            self.assertTrue(torch.allclose(fresh.fuser.voice_proj.bias,
                                           torch.full((2,), 0.42)))
            fresh.enroll("meeting-a", torch.ones(3), torch.ones(4))
            self.assertEqual(fresh.speaker_ids, ("meeting-a",))
            normal = IdentityPool(self.cfg)
            normal.load(path)
            self.assertEqual(normal.speaker_ids, ("train-speaker",))
            voice_only = normal.query(torch.ones(3), None)
            self.assertEqual(voice_only.evidence_mode, "voice_only")
            self.assertEqual(voice_only.top_ids, ["train-speaker"])


class AttentionAndGenerationTest(unittest.TestCase):
    def test_qwen_and_llama_generation_receive_explicit_aligned_mask(self):
        for family, hidden in (("qwen2.5-3b-instruct", 2048),
                               ("llama-3.2-3b-instruct", 3072)):
            cfg = ger_cfg(family)
            backend = CapturingBackend(cfg)
            head = GERHead(cfg, z_dim=8, d_align=16, backend=backend)
            head.generate(torch.zeros(8), torch.zeros(3, 16), ["hello"])
            kwargs = backend.model.kwargs
            self.assertEqual(kwargs["inputs_embeds"].shape[-1], hidden)
            self.assertEqual(kwargs["attention_mask"].shape,
                             kwargs["inputs_embeds"].shape[:2])
            self.assertEqual(kwargs["attention_mask"].dtype, torch.long)
            self.assertNotIn("temperature", kwargs)
            self.assertNotIn("top_p", kwargs)
            self.assertNotIn("top_k", kwargs)

    def test_tokenizer_padding_mask_is_preserved_even_when_pad_equals_eos(self):
        cfg = ger_cfg()
        head = GERHead(cfg, z_dim=8, d_align=16, backend=CapturingBackend(cfg))
        original = head._tok.__call__

        def encoded(text, **kwargs):
            result = original(text, **kwargs)
            result.attention_mask = torch.ones_like(result.input_ids)
            if result.input_ids.numel():
                result.attention_mask[0, -1] = 0
            return result

        head._tok.__call__ = encoded  # special lookup ignores instance override
        # Patch class lookup used by Python's special method dispatch.
        with patch.object(FakeTokenizer, "__call__", side_effect=encoded):
            prompt = head._render_text(None, ["hello"], "")
            embeds, mask = head._model_inputs(torch.zeros(8), torch.zeros(3, 16), prompt)
        self.assertEqual(mask.shape, embeds.shape[:2])
        self.assertIn(0, mask.tolist()[0])

    def test_batched_padding_stays_aligned_with_soft_tokens(self):
        cfg = ger_cfg()
        head = GERHead(cfg, z_dim=8, d_align=16, backend=CapturingBackend(cfg))
        original = head._tok.__call__

        def batched(text, **kwargs):
            one = original(text, **kwargs)
            ids = one.input_ids.repeat(2, 1)
            mask = torch.ones_like(ids)
            if ids.shape[1]:
                ids[1, -1] = head._tok.pad_token_id
                mask[1, -1] = 0
            return SimpleNamespace(input_ids=ids, attention_mask=mask)

        with patch.object(FakeTokenizer, "__call__", side_effect=batched):
            prompt = head._render_text("speaker", ["hello"], "visual")
            embeds, mask = head._model_inputs(
                torch.zeros(2, 8), torch.zeros(2, 3, 16), prompt
            )
        self.assertEqual(embeds.shape[:2], mask.shape)
        self.assertEqual(embeds.shape[0], 2)
        self.assertGreater((mask[1] == 0).sum().item(), 0)

    def test_sampling_parameters_exist_only_when_sampling_enabled(self):
        from avsd_ger.c2_alignment.generation_policy import GenerationPolicy
        deterministic = GenerationPolicy(do_sample=False, temperature=0.7,
                                         top_p=0.8, top_k=20)
        self.assertEqual(set(deterministic.kwargs(1)) & {"temperature", "top_p", "top_k"}, set())
        sampling = GenerationPolicy(do_sample=True, temperature=0.7,
                                    top_p=0.8, top_k=20)
        self.assertEqual(sampling.kwargs(1)["top_k"], 20)

    def test_training_passes_mask_and_labels_only_the_target_span(self):
        from avsd_ger.training.ger_loss import GERCrossEntropy
        cfg = ger_cfg()
        backend = CapturingBackend(cfg)
        head = GERHead(cfg, z_dim=8, d_align=16, backend=backend)
        report = GERCrossEntropy(head)(
            torch.zeros(8), torch.zeros(3, 16), ["helo"], "", "hello"
        )
        self.assertGreater(report.n_target_tokens, 0)
        self.assertEqual(backend.model.kwargs["attention_mask"].shape,
                         backend.model.kwargs["inputs_embeds"].shape[:2])


class CheckpointAndDtypeTest(unittest.TestCase):
    def test_strict_metadata_and_legacy_opt_in(self):
        head = GERHead(ger_cfg(), z_dim=8, d_align=16, stub=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good.pt"
            head.save_projector_checkpoint(good)
            state = torch.load(good, weights_only=True)
            legacy = root / "legacy.pt"
            state.pop("metadata")
            torch.save(state, legacy)
            with self.assertRaisesRegex(CheckpointCompatibilityError, "no metadata"):
                head.load_projector_checkpoint(legacy)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                head.load_projector_checkpoint(legacy, allow_legacy=True)
            self.assertIn("UNSAFE LEGACY OPT-IN", str(caught[0].message))

    def test_bridge_heads_lora_and_chat_template_mismatches_are_rejected(self):
        saved = GERHead(ger_cfg(), z_dim=8, d_align=16, stub=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.pt"
            saved.save_projector_checkpoint(path)
            for current, expected in (
                (GERHead(ger_cfg(heads=4), z_dim=8, d_align=16, stub=True), "n_heads"),
                (GERHead({**ger_cfg(), "lora": {**ger_cfg()["lora"], "r": 8}}, z_dim=8, d_align=16, stub=True), "lora_rank"),
            ):
                with self.assertRaisesRegex(CheckpointCompatibilityError, expected):
                    current.load_projector_checkpoint(path)
            changed = GERHead(ger_cfg(), z_dim=8, d_align=16, stub=True)
            changed._tok.chat_template = "different"
            with self.assertRaisesRegex(CheckpointCompatibilityError, "chat_template"):
                changed.load_projector_checkpoint(path)

    def test_all_metadata_corruption_classes_fail_before_weights_load(self):
        head = GERHead(ger_cfg(), z_dim=8, d_align=16, stub=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good = root / "good.pt"
            head.save_projector_checkpoint(good)
            original = torch.load(good, weights_only=True)
            mutations = {
                "hidden_size": ("hidden_size", 999),
                "tokenizer_vocab_size": ("tokenizer_vocab_size", 999),
                "lora_target_modules": ("lora_target_modules", ["q_proj"]),
            }
            for label, (field, value) in mutations.items():
                state = {**original, "metadata": dict(original["metadata"])}
                state["metadata"][field] = value
                path = root / f"{label}.pt"
                torch.save(state, path)
                with self.subTest(label=label), self.assertRaisesRegex(
                    CheckpointCompatibilityError, field
                ):
                    head.load_projector_checkpoint(path)
            corrupt = root / "corrupt.pt"
            torch.save({**original, "metadata": "broken"}, corrupt)
            with self.assertRaisesRegex(CheckpointCompatibilityError, "not a mapping"):
                head.load_projector_checkpoint(corrupt)

    def test_dtype_auto_checks_cuda_bf16_capability(self):
        cuda = torch.device("cuda")
        with patch("torch.cuda.is_bf16_supported", return_value=True):
            self.assertIs(LocalHFCausalLMBackend._resolve_dtype("auto", cuda), torch.bfloat16)
        with patch("torch.cuda.is_bf16_supported", return_value=False):
            self.assertIs(LocalHFCausalLMBackend._resolve_dtype("auto", cuda), torch.float16)
            with self.assertRaisesRegex(ValueError, "does not support"):
                LocalHFCausalLMBackend._resolve_dtype("bf16", cuda)
        self.assertIs(LocalHFCausalLMBackend._resolve_dtype("auto", torch.device("cpu")), torch.float32)


class ConfigAndSafetyTest(unittest.TestCase):
    def test_public_inheritance_and_local_only_policy(self):
        qwen = load_config("configs/qwen25_3b.yaml")
        llama = load_config("configs/llama32_3b.yaml")
        short = load_config("configs/qwen25_3b_short.yaml")
        self.assertFalse(qwen["ger"]["allow_download"])
        self.assertFalse(llama["ger"]["allow_download"])
        self.assertEqual(short["ger"]["max_new_tokens"], 24)
        self.assertEqual(short["identity"], qwen["identity"])

    def test_unknown_inherited_key_fails_and_lists_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.yaml").write_text("x:\n  values: [a, b]\n", encoding="utf-8")
            (root / "good.yaml").write_text("defaults: [base]\nx:\n  values: [c]\n", encoding="utf-8")
            (root / "bad.yaml").write_text("defaults: [base]\nx:\n  typo: 1\n", encoding="utf-8")
            self.assertEqual(load_config(root / "good.yaml")["x"]["values"], ["c"])
            with self.assertRaisesRegex(KeyError, "x.typo"):
                load_config(root / "bad.yaml")

    def test_safety_rejects_degradation_and_exposes_every_gate(self):
        gate = GERSafetyGate()
        cases = (
            ("", "hello world", "empty"),
            ("hello um world", "hello world", "filler"),
            ("hello world hello world hello world", "hello world", "repeated"),
            ("bonjour complètement différent", "hello world", "overlap"),
            ("hello <|assistant|> world", "hello world", "artifact"),
            ("we met Bob today", "we met Alice today", "person name"),
        )
        for ger, asr, expected in cases:
            with self.subTest(expected=expected):
                result = gate.evaluate(ger, asr)
                self.assertFalse(result.accepted)
                self.assertIn(expected.split()[0].lower(), result.reason.lower())
                self.assertTrue(result.checks)
        high = gate.evaluate("please close the red door", "please close the door",
                             asr_confidence=0.95)
        self.assertFalse(high.accepted)
        self.assertIn("High-confidence", high.reason)
        self.assertTrue(gate.evaluate("please close the door", "please close door").accepted)


if __name__ == "__main__":
    unittest.main()
