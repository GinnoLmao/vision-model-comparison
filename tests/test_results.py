"""Dependency-free checks on historical artifacts and portable notebook state."""
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_results import verify_results


class HistoricalArtifactChecks(unittest.TestCase):
    def test_six_recorded_scores_agree_with_confusion_counts(self):
        data, rows = verify_results(ROOT / "results/model_results.json")
        self.assertEqual(len(rows), 6)
        self.assertEqual(data["dataset"]["classes"], ["ants", "bees"])
        self.assertEqual(sum(row["validation_accuracy"] == 148 / 153 for row in rows), 3)

    def test_notebook_has_no_execution_outputs_or_absolute_user_paths(self):
        path = ROOT / "notebooks/six_model_comparison.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            self.assertFalse(re.search(r"[A-Z]:[/\\]+Users[/\\]+", source))
            if cell["cell_type"] == "code":
                compile(source, "notebook cell", "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])


if __name__ == "__main__":
    unittest.main()
