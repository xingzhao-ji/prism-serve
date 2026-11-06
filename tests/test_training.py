import numpy as np
import torch

from prismserve.experiment import train_head


def test_training_and_selection_ignore_all_test_features_and_labels():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(60, 512)).astype("float32")
    labels = np.arange(60) % 10
    splits = np.array(["train"] * 30 + ["validation"] * 20 + ["test"] * 10)
    first, report = train_head(features, labels, splits, epochs=3)
    features[splits == "test"] = 999
    labels[splits == "test"] = 9
    second, changed = train_head(features, labels, splits, epochs=3)
    assert report == changed
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_exported_affine_head_matches_standardized_training_predictions():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(40, 512)).astype("float32")
    labels = np.arange(40) % 10
    splits = np.array(["train"] * 20 + ["validation"] * 20)
    head, report = train_head(features, labels, splits, epochs=3)
    assert head["weight"].shape == (10, 512)
    assert np.isfinite(head["weight"]).all()
    assert report["folded_max_error"] < 1e-4
    logits = torch.tensor(features) @ torch.tensor(head["weight"]).T + torch.tensor(head["bias"])
    assert logits.shape == (40, 10)
