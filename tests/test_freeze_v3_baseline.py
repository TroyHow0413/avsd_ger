import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.freeze_v3_baseline import _checkpoint_metadata, _inventory, freeze


class BaselineFreezeTest(unittest.TestCase):
    def test_inventory_is_stable_and_content_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.txt").write_text("b", encoding="utf-8")
            (root / "a.txt").write_text("a", encoding="utf-8")
            first = _inventory(root)
            second = _inventory(root)
            self.assertEqual(first["inventory_sha256"], second["inventory_sha256"])
            self.assertEqual(first["file_count"], 2)
            (root / "a.txt").write_text("changed", encoding="utf-8")
            self.assertNotEqual(
                first["inventory_sha256"], _inventory(root)["inventory_sha256"]
            )

    def test_source_destination_overlap_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            with self.assertRaises(ValueError):
                freeze(source, source / "out", source / "provenance.json")

    def test_provenance_cannot_be_written_into_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            with self.assertRaisesRegex(ValueError, "provenance output"):
                freeze(
                    source,
                    Path(__file__).resolve().parents[1] / "out" / "baselines" / "test",
                    source / "provenance.json",
                )

    def test_checkpoint_metadata_is_extracted_without_tensor_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "checkpoint/data.pkl",
                    pickle.dumps({"metadata": {"model_family": "qwen", "hidden_size": 2048}}),
                )
            result = _checkpoint_metadata(path)
        self.assertEqual(result["status"], "extracted")
        self.assertEqual(result["value"]["hidden_size"], 2048)


if __name__ == "__main__":
    unittest.main()
