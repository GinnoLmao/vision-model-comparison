# Validation record — 21 November 2026

**Eight tests passed, with no skips**, in the shared isolated Python 3.12.10 CPU environment. Tested libraries: PyTorch 2.15.0, TorchVision 0.29.2, NumPy 2.5.5, pandas 3.1.1, Matplotlib 3.11.2, scikit-learn 1.9.1 and nbformat 5.11.1. CPU tests used two threads, batch size two and `weights=None`; no pretrained weights were downloaded.

## Executed

`python -m unittest discover -s tests -v` passed all eight methods:

- All six architectures return finite two-class logits with the intended 128/224-pixel input dimensions.
- Frozen encoders remain in evaluation mode during head training; partial fine-tuning enables the intended final block.
- A synthetic head-optimizer step leaves frozen batch-normalization running statistics unchanged and produces head gradients without backbone gradients.
- A single two-sample scratch-CNN epoch exercises the training/validation loop and restores its selected checkpoint. This is a software smoke check, not a new study result.
- Validation transforms are deterministic and preserve expected image shapes; invalid model/recurrent names are rejected.
- Historical artifacts agree with confusion counts, and the portable notebook has no execution outputs or absolute user paths.

Additional checks:

- Formal nbformat validation passed for all 38 notebook cells; all Python source and notebook code cells parsed successfully.
- All six historical confusion matrices sum to 153; accuracy and macro precision/recall/F1 recompute consistently with saved summaries (tolerance 1e-8).
- The publication JSON's entire `models` and `settings` objects match the original source artifact. Only its absolute dataset path and a publication note changed.
- Original notebook outputs were cleared, including sample photographs and machine paths. The metrics-only chart was regenerated with the supplied Matplotlib script and visually checked; it contains no dataset images.
- Read-only source-dataset audit: train ants 123, train bees 121, validation ants 70, validation bees 83; zero byte-identical hashes shared between partitions. This does not establish near-duplicate or subject-level independence.
- Original vision notebook/results byte hashes still matched preparation provenance; no original project files were changed.
- An additional end-to-end CLI smoke run used eight newly generated solid-color PNGs (four training/four validation), `CNN2D`, random initialization, one CPU epoch and two threads. ImageFolder loading, augmentation, fitting, validation and new-run JSON export completed successfully. Temporary images were removed; the ignored smoke output is not a study result and is not published.

## Not executed or not established

- No historical GPU experiment was retrained, and original checkpoints were not loaded or published.
- Random-initialization architecture checks do not reproduce learned ImageNet features or historical model accuracy.
- No independent test set, external-domain evaluation, repeated-seed uncertainty estimate or significance comparison was established.
- No rights to redistribute dataset photographs or pretrained artifacts were inferred from a tutorial's code license.

## Implementation changes and scope

Portable dataset selection, repository-local non-overwriting run paths, a frozen recurrent-backbone `train()` override, invalid-name guards and explicit deterministic settings were added to the reusable entry point. Historical result numbers were retained and are not relabeled as new performance.

The reused validation set cannot become an untouched test set retrospectively. The saved scores remain descriptive validation evidence; a future independent evaluation needs a new prespecified protocol and new results.
