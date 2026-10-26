"""A3-owned synthetic Food-101 fixture, independent of the baseline tests."""
import json

import pytest
from PIL import Image


@pytest.fixture
def food_fixture(tmp_path):
    root = tmp_path / "food-101"
    (root / "meta").mkdir(parents=True)
    manifests = {"train": {}, "test": {}}
    for label in range(101):
        name = f"class_{label:03d}"
        (root / "images" / name).mkdir(parents=True)
        for split, count in (("train", 3), ("test", 1)):
            manifests[split][name] = []
            for i in range(count):
                relative = f"{name}/{split}_{i}"
                Image.new("RGB", (40, 40), (label * 2, 30 + i * 40, 80)).save(
                    root / "images" / f"{relative}.jpg")
                manifests[split][name].append(relative)
    for split, manifest in manifests.items():
        (root / "meta" / f"{split}.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path
