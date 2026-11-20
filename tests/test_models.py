"""Offline checks: synthetic tensors/PIL images, weights=None, no training claims."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import numpy as np
    from PIL import Image
    import torch
    from vision_comparison import MODEL_NAMES, build_model, make_transforms, set_training_mode
except ImportError as exc:
    raise unittest.SkipTest("Architecture checks require installed torch/torchvision/numpy/Pillow; no download is attempted") from exc


class ArchitectureChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_six_models_return_two_logits_without_downloads(self):
        for name in MODEL_NAMES:
            with self.subTest(name=name):
                model, mode = build_model(name, weights=None)
                set_training_mode(model, False, mode)
                size = 128 if mode is None else 224
                with torch.inference_mode():
                    logits = model(torch.zeros(2, 3, size, size))
                self.assertEqual(tuple(logits.shape), (2, 2))
                self.assertTrue(torch.isfinite(logits).all())

    def test_frozen_backbone_statistics_stay_in_eval_mode(self):
        for name in MODEL_NAMES[2:]:
            model, mode = build_model(name, weights=None)
            set_training_mode(model, True, mode)
            if mode == "sequence":
                model.train()  # The class override also protects direct callers.
                self.assertFalse(model.backbone.training)
                self.assertTrue(model.recurrent.training)
                self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
            else:
                self.assertFalse(model.bn1.training)
                self.assertTrue(model.fc.training)
                self.assertEqual(model.layer4.training, mode == "partial")

    def test_validation_transforms_are_deterministic_and_match_sizes(self):
        image = Image.fromarray(np.arange(300 * 270 * 3, dtype=np.uint8).reshape(300, 270, 3))
        for size in (128, 224):
            train, validation = make_transforms(size)
            a, b = validation(image), validation(image)
            self.assertTrue(torch.equal(a, b))
            self.assertEqual(tuple(a.shape), (3, size, size))
            self.assertEqual(tuple(train(image).shape), (3, size, size))

    def test_invalid_model_and_recurrent_kind_are_rejected(self):
        from vision_comparison import PretrainedRowSequence
        with self.assertRaises(ValueError):
            build_model("typo", weights=None)
        with self.assertRaises(ValueError):
            PretrainedRowSequence("typo", weights=None)

    def test_head_step_does_not_change_frozen_backbone_statistics(self):
        model, mode = build_model("ResNet18 feature extractor", weights=None)
        set_training_mode(model, True, mode)
        before = model.bn1.running_mean.clone()
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=.01)
        logits = model(torch.ones(2, 3, 224, 224))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        optimizer.step()
        self.assertTrue(torch.equal(before, model.bn1.running_mean))
        self.assertIsNone(model.conv1.weight.grad)
        self.assertIsNotNone(model.fc[-1].weight.grad)

    def test_synthetic_single_epoch_restores_selected_checkpoint(self):
        from torch.utils.data import DataLoader, TensorDataset
        from vision_comparison import fit, make_optimizer
        model, mode = build_model("CNN2D", weights=None)
        tensor_data = TensorDataset(torch.zeros(2, 3, 128, 128), torch.tensor([0, 1]))
        loaders = {phase: DataLoader(tensor_data, batch_size=2) for phase in ("train", "val")}
        trained, history, best_epoch = fit(model, loaders, make_optimizer(model, mode), 1, "cpu", mode)
        self.assertEqual(best_epoch, 1)
        self.assertEqual(len(history), 1)
        self.assertFalse(trained.training)
        self.assertTrue(np.isfinite(history[0]["val_loss"]))


if __name__ == "__main__":
    unittest.main()
