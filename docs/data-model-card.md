# Data and model card

The input corpus is [fastai Imagenette](https://github.com/fastai/imagenette), the 160-pixel version with preserved aspect ratio. It contains ten ImageNet synsets. The project downloads the official archive and pretrained [torchvision ResNet18 weights](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py), verifying complete hashes:

| Artifact | SHA-256 |
|---|---|
| `imagenette2-160.tgz` | `64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5` |
| `resnet18-f37072fd.pth` | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |

Hashes were calculated from these public downloads during preparation. They detect later byte changes; they are not signatures or independent security review. Raw images and weights are not redistributed in the source repository. Imagenette's source images inherit ImageNet/source image rights; this project does not assert ownership or a blanket commercial-use license. Follow the original dataset's terms for further use. Repository code licensing does not license the images or pretrained weights.

Each class is sorted by SHA-256 of its original filename. The first 80 training-source images are assigned 60 to training and 20 to selection; the first 30 official-validation images go to assessment. No model predictions influence selection. The committed subset manifest records every path, label, dimension, raw hash and decoded-pixel hash. Integrity is rechecked before feature extraction/evaluation. Archive extraction rejects traversal, links, non-file entries and excessive total size.

QA covers selected-image decoding, folder-derived class mapping, exact byte duplicates and exact decoded-pixel duplicates. It does not prove absence of similar crops, relabeled duplicates, errors in original labels or correlated subjects. No person identity labels are inferred. The 10 classes are balanced by construction; deployment prevalence and out-of-scope objects are not modeled.

The backbone is frozen and eval-mode throughout feature extraction. A 512→10 linear head is trained with AdamW and standardization fit only on training rows. Weight decay is selected from `0, 0.001, 0.01` using selection accuracy, then cross-entropy. Export folds standardization into an affine weight matrix/bias, with a numerical equivalence check. A mutation test changes all assessment embeddings and labels and requires identical trained parameters and selection reports.

Assessment is separated from selection in the CLI. Outputs are immutable by default; reruns need new directories. The evaluation compares unrestricted ImageNet output, restricted known-class pretrained logits, the learned head, and a predeclared Gaussian blur slice. The head fails the measured improvement gate and is optional in serving. The bootstrap interval is paired by image and assumes independent images; source correlations and a small 300-image sample limit its interpretation.

The source training images overlap pretrained ImageNet training; official validation is not an untouched source for the pretrained model family. Consequently this project demonstrates transfer-learning mechanics and controlled local assessment, not new out-of-distribution generalization, production readiness, data collection expertise or end-to-end deep backbone training.
