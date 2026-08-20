import tempfile
import unittest
from pathlib import Path

from avsd_ger.utils import load_config


class ConfigLoadingTest(unittest.TestCase):
    def test_merges_nested_defaults_without_mutating_base_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "base.yaml").write_text(
                "model:\n  width: 64\n  depth: 4\nseed: 1\n", encoding="utf-8"
            )
            (root / "child.yaml").write_text(
                "defaults: [base]\nmodel:\n  depth: 8\n", encoding="utf-8"
            )

            cfg = load_config(root / "child.yaml")

            self.assertEqual(cfg["model"], {"width": 64, "depth": 8})
            self.assertEqual(cfg["seed"], 1)

    def test_empty_config_is_an_empty_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.yaml"
            path.write_text("", encoding="utf-8")

            self.assertEqual(load_config(path), {})

    def test_rejects_non_mapping_root(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "list.yaml"
            path.write_text("- one\n- two\n", encoding="utf-8")

            with self.assertRaisesRegex(TypeError, "root must be a mapping"):
                load_config(path)

    def test_detects_defaults_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.yaml").write_text("defaults: [b]\n", encoding="utf-8")
            (root / "b.yaml").write_text("defaults: [a]\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "cycle detected"):
                load_config(root / "a.yaml")


if __name__ == "__main__":
    unittest.main()
