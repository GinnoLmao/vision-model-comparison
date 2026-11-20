"""Optional retraining. Explicit --weights imagenet may download official weights."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from vision_comparison import MODEL_NAMES, build_model, make_loaders, make_optimizer, fit, seed_all

SEEDS = {"CNN2D": 142, "CNN3D (RGB depth)": 143, "RNN (ResNet features)": 142,
         "LSTM (ResNet features)": 143, "ResNet18 feature extractor": 144, "ResNet18 partial fine-tune": 145}
EPOCHS = {"CNN2D": 28, "CNN3D (RGB depth)": 28, "RNN (ResNet features)": 14,
          "LSTM (ResNet features)": 14, "ResNet18 feature extractor": 16, "ResNet18 partial fine-tune": 14}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models", choices=MODEL_NAMES, nargs="+", default=list(MODEL_NAMES))
    parser.add_argument("--weights", choices=("imagenet", "none"), default="imagenet")
    parser.add_argument("--epochs", type=int, help="Override schedules for a smoke run; this changes the experiment")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.set_num_threads(2)
    output = ROOT / "results/runs" / time.strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    payload = {"evaluation": "validation selected checkpoint and supplied final scores; no independent test", "weights": args.weights,
               "deterministic_requested": True, "models": {}}
    for name in args.models:
        seed_all(SEEDS[name])
        model, freeze_mode = build_model(name, None if args.weights == "none" else "imagenet")
        model.to(args.device)
        loaders = make_loaders(args.data_dir, 128 if freeze_mode is None else 224, device=args.device)
        optimizer = make_optimizer(model, freeze_mode)
        model, history, best_epoch = fit(model, loaders, optimizer, args.epochs or EPOCHS[name], args.device, freeze_mode)
        truth, predictions = [], []
        with torch.inference_mode():
            for images, labels in loaders["val"]:
                predictions.extend(model(images.to(args.device)).argmax(1).cpu().tolist())
                truth.extend(labels.tolist())
        precision, recall, f1, _ = precision_recall_fscore_support(truth, predictions, average="macro", zero_division=0)
        payload["models"][name] = {"seed": SEEDS[name], "epochs": args.epochs or EPOCHS[name], "best_epoch": best_epoch,
                                   "validation_samples": len(truth), "validation_accuracy": accuracy_score(truth, predictions),
                                   "macro_precision": precision, "macro_recall": recall, "macro_f1": f1,
                                   "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(), "history": history}
        (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: saved validation results to {output.name}/results.json")


if __name__ == "__main__":
    main()
