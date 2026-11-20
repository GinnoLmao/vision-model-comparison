"""Verify and summarize saved validation artifacts without images or checkpoints."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify_results(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert len(data["models"]) == 6
    n = data["dataset"]["validation_count"]
    assert n == 153 and data["dataset"]["train_count"] == 244
    rows = []
    for name, result in data["models"].items():
        cm = result["confusion_matrix"]
        assert len(cm) == 2 and all(len(row) == 2 for row in cm)
        assert all(isinstance(x, int) and x >= 0 for row in cm for x in row)
        assert sum(map(sum, cm)) == n
        accuracy = (cm[0][0] + cm[1][1]) / n
        precision = [cm[i][i] / sum(cm[j][i] for j in range(2)) for i in range(2)]
        recall = [cm[i][i] / sum(cm[i]) for i in range(2)]
        f1 = [2 * p * r / (p + r) if p + r else 0 for p, r in zip(precision, recall)]
        summary = result["summary"]
        expected = {"Accuracy": accuracy, "Precision (macro)": sum(precision) / 2,
                    "Recall (macro)": sum(recall) / 2, "F1 (macro)": sum(f1) / 2}
        for metric, value in expected.items():
            assert math.isclose(summary[metric], value, abs_tol=1e-8), (name, metric)
        assert summary["Validation samples"] == n
        assert summary["Best epoch"] in [row["epoch"] for row in result["history"]]
        rows.append({"model": name, "validation_accuracy": accuracy,
                     "macro_f1": sum(f1) / 2, "best_epoch": summary["Best epoch"], "n_validation": n})
    return data, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="Create a chart of historical metrics only")
    args = parser.parse_args()
    data, rows = verify_results(ROOT / "results/model_results.json")
    with (ROOT / "results/validation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Verified {len(rows)} saved confusion matrices and macro metrics; validation n=153, no independent test.")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ranked = sorted(rows, key=lambda row: row["validation_accuracy"])
        fig, ax = plt.subplots(figsize=(10, 4.7))
        values = [row["validation_accuracy"] for row in ranked]
        ax.barh([row["model"] for row in ranked], values, color="#244c67")
        for index, value in enumerate(values):
            ax.text(value + .008, index, f"{value:.1%}", va="center")
        ax.set(xlim=(0, 1.08), xlabel="Saved validation accuracy", title="Historical six-model validation comparison")
        ax.grid(axis="x", alpha=.2)
        fig.text(.5, .012, "153 reused validation images; checkpoint/model selection used these labels. No untouched test set.", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(ROOT / "results/validation_comparison.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
