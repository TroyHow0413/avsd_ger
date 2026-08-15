from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import torch  # noqa: F401
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is required")
class PipelineMouthROILazyTest(unittest.TestCase):
    def _stub_config(self):
        from avsd_ger.utils import load_config

        cfg = load_config("configs/default.yaml")
        cfg["device"] = "cpu"
        cfg["stub_backbones"] = True
        cfg["ger"]["mode"] = "av"
        cfg["mouth_roi"]["backend"] = "dlib"
        cfg["alignment"].update({"d_model": 16, "n_heads": 4, "n_layers": 1})
        cfg["identity"]["fused_dim"] = 8
        return cfg

    def test_precomputed_roi_path_does_not_construct_dlib(self):
        from avsd_ger.pipeline import AVSDGERPipeline

        with patch(
            "avsd_ger.pipeline.MouthROIExtractor",
            side_effect=AssertionError("native mouth ROI extractor must stay lazy"),
        ):
            pipe = AVSDGERPipeline(self._stub_config())

        self.assertIsNone(pipe.mouth_roi_extractor)

    def test_raw_video_path_constructs_configured_backend_on_demand(self):
        from avsd_ger.pipeline import AVSDGERPipeline

        pipe = AVSDGERPipeline(self._stub_config())
        sentinel = object()
        with patch("avsd_ger.pipeline.MouthROIExtractor", return_value=sentinel) as ctor:
            self.assertIs(pipe._get_mouth_roi_extractor(), sentinel)
            self.assertIs(pipe._get_mouth_roi_extractor(), sentinel)

        ctor.assert_called_once()
        self.assertEqual(ctor.call_args.kwargs["backend"], "dlib")


if __name__ == "__main__":
    unittest.main()
