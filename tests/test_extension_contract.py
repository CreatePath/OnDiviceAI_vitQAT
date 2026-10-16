"""Adding experiment behavior must not require edits to the common runner."""
import ast
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from vitqat.cli import parser
from vitqat.runner import run
from vitqat.training import evaluate, train_epoch


def test_runner_accepts_independent_model_training_data_and_metrics(tmp_path):
    torch.set_num_threads(2)
    args = parser().parse_args([
        "--mode", "fp32", "--experiment", "new-experiment",
        "--output", str(tmp_path / "new"), "--device", "cpu", "--epochs", "1",
        "--warmup-epochs", "0", "--workers", "0", "--batch-size", "2",
    ])
    calls = []

    def make_model(config):
        calls.append("model")
        return nn.Linear(4, 2)  # No timm-specific methods or QAT attributes.

    def make_dataset(config):
        calls.append("data")
        data = TensorDataset(torch.randn(4, 4), torch.tensor([0, 1, 0, 1]))
        return data, data, data, {"sha256": "fixture-only"}

    def custom_train(*arguments):
        calls.append("train")
        return train_epoch(*arguments)

    def custom_eval(*arguments):
        calls.append("evaluate")
        return evaluate(*arguments)

    def custom_metrics(model, config):
        calls.append("metrics")
        return {"token_pruning_ratio": 0.5, "tokens_per_block": [5, 3], "gflops": 0.123,
                "params": sum(p.numel() for p in model.parameters()), "custom_measurement": 42}

    metrics = run(args, build_model=make_model, measure_model=custom_metrics,
                  data_factory=make_dataset, train_epoch_fn=custom_train, evaluate_fn=custom_eval)
    assert calls == ["data", "model", "train", "evaluate", "evaluate", "metrics"]
    assert metrics["experiment"] == "new-experiment"
    assert metrics["token_pruning_ratio"] == 0.5 and metrics["tokens_per_block"] == [5, 3]
    assert metrics["gflops"] == 0.123 and metrics["custom_measurement"] == 42


def test_shared_code_does_not_import_experiments():
    root = Path(__file__).resolve().parents[1] / "vitqat"
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith("experiments") for alias in node.names), path
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("experiments"), path
