import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None


if torch is None:
    @unittest.skip("torch is required for pipeline tests")
    class PipelineAudioOnlyMaskTest(unittest.TestCase):
        def test_audio_only_ignores_visual_length_masks(self):
            pass
else:
    from avsd_ger.pipeline import AVSDGERPipeline
    from avsd_ger.utils import load_config


    class PipelineAudioOnlyMaskTest(unittest.TestCase):
        def test_audio_only_ignores_visual_length_masks(self):
            cfg = load_config("configs/default.yaml")
            cfg["device"] = "cpu"
            cfg["stub_backbones"] = True
            cfg["ger"]["mode"] = "audio_only"
            cfg["mouth_roi"]["backend"] = "haar"
            cfg["alignment"].update({"d_model": 16, "n_heads": 4, "n_layers": 1})
            cfg["identity"]["fused_dim"] = 8

            pipe = AVSDGERPipeline(cfg)
            out = pipe.run(
                audio_wav=torch.zeros(16000),
                video_frames=None,
                has_visual=False,
                speaker_mask_v=torch.ones(271, dtype=torch.bool),
                lip_conf_v=torch.ones(271),
            )

            self.assertEqual(out["debug"]["visual"]["effective_ger_mode"], "audio_only")
            self.assertFalse(
                out["trace"][-1]["alignment"]["speaker_mask_v_present"]
            )
            self.assertFalse(
                out["trace"][-1]["alignment"]["lip_conf_v_present"]
            )


if __name__ == "__main__":
    unittest.main()
