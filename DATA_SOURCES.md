# Data and pretrained weights

The original experiment uses the ants/bees **Hymenoptera** archive linked by the official [PyTorch transfer-learning tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). The tutorial identifies the data as a small subset of ImageNet and links [the archive on download.pytorch.org](https://download.pytorch.org/tutorial/hymenoptera_data.zip). These primary sources were reviewed on 20 November 2026.

The observed local partition contains 123 ants and 121 bees for training, and 70 ants and 83 bees for validation. These counts describe the supplied files; they are not a claim of a newly collected dataset.

**A tutorial code license does not establish rights to redistribute its photographs.** No per-image redistribution grant was established in the inspected material. This repository therefore excludes all photographs, the dataset archive, checkpoints and sample-photo notebook outputs. Users should acquire data through the official tutorial route and review the source dataset's applicable terms themselves. The [ImageNet access page](https://www.image-net.org/download.php) is provided as a provenance reference, not as evidence that permission was obtained here; it could not be retrieved during this preparation.

The four ResNet-based models initialize from TorchVision's ImageNet ResNet18 weights. No pretrained or fine-tuned weights are bundled. Consult the [TorchVision ResNet18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) and its linked provenance before using or redistributing weights. The architecture tests explicitly use random initialization and do not access the network.

The compact result JSON contains aggregate numerical metrics, histories and confusion counts, not image data. The chart is generated solely from those saved metrics. The source artifact's absolute machine path was removed; model numbers were preserved exactly.
