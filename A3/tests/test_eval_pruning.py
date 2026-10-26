import json

import pytest
import torch

from vitqat.data import make_data
from A3.eval_pruning import (inherited_args, load_router_bundle, parser, run,
                                 sha256_file, validate_qat_state)
from A3.inference_pruning import PruningConfig
from vitqat.training import save_checkpoint
from A3.tests.test_inference_pruning import fixture_queries, settings, small_fixture_model


@pytest.fixture
def inputs(food_fixture):
    torch.set_num_threads(2)
    args = settings(food_fixture)
    args.data_root = str(food_fixture)
    args.experiment = "upstream-owned-name"
    # Deliberately non-default inherited settings must survive evaluation.
    args.lr, args.epochs, args.label_smoothing = 0.000123, 50, 0.2
    args.weight_bits, args.activation_bits = 2, 8
    model = small_fixture_model(args)
    with torch.no_grad():
        model(torch.randn(2, 3, 224, 224))
    _, _, _, manifest = make_data(args)
    checkpoint = food_fixture / "best.pt"
    save_checkpoint(checkpoint, model, torch.optim.AdamW(model.parameters()), 17, 42., args, manifest["sha256"])
    router = food_fixture / "queries.pt"
    bundle = dict(format_version=1, queries_are_trained=True,
                  backbone_sha256=sha256_file(checkpoint), split_hash=manifest["sha256"],
                  router_config=PruningConfig().describe(12, 196), queries=fixture_queries(),
                  training=dict(method="synthetic-test-fixture-only", data="generated fixture",
                                backbone_frozen=True, observers_frozen=True, pruning_during_training=True))
    torch.save(bundle, router)
    options = parser().parse_args(["--checkpoint", str(checkpoint), "--router-checkpoint", str(router),
                                  "--output", str(food_fixture / "pair"), "--device", "cpu",
                                  "--workers", "0", "--threads", "2", "--eval-batch-size", "32",
                                  "--latency-warmup", "0", "--latency-repeats", "1"])
    return args, model, manifest, bundle, options


def test_paired_pipeline_inherits_config_and_preserves_inputs(inputs):
    args, model, manifest, bundle, options = inputs
    before = (sha256_file(options.checkpoint), sha256_file(options.router_checkpoint))
    result = run(options, model_factory=small_fixture_model)
    assert result["backbone_state_unchanged"] and result["router_state_unchanged"]
    from pathlib import Path
    output = Path(options.output)
    config = json.loads((output / "evaluation_config.json").read_text())
    assert config["training_config"]["lr"] == args.lr
    assert config["effective_config"]["epochs"] == 50
    assert config["effective_config"]["label_smoothing"] == 0.2
    assert config["effective_config"]["experiment"] == "upstream-owned-name"
    assert config["effective_config"]["weight_bits"] == 2
    assert config["effective_config"]["batch_size"] == 32
    dense = json.loads((output / "dense_metrics.json").read_text())
    pruned = json.loads((output / "pruned_metrics.json").read_text())
    assert dense["checkpoint_sha256"] == pruned["checkpoint_sha256"] == before[0]
    assert dense["split_hash"] == pruned["split_hash"] == manifest["sha256"]
    assert dense["test_samples"] == pruned["test_samples"] == 101
    assert dense["token_pruning_ratio"] == 0
    assert pruned["tokens_per_block"] == [197,180,164,148,132,116,100,84,68,52,36,197]
    assert pruned["params"] - dense["params"] == 120
    assert result["top1_loss_pp"] == dense["top1_acc"] - pruned["top1_acc"]
    assert (output / "comparison.csv").exists()
    assert before == (sha256_file(options.checkpoint), sha256_file(options.router_checkpoint))
    with pytest.raises(ValueError, match="empty"):
        run(options, model_factory=small_fixture_model)


@pytest.mark.parametrize("field,value,match", [
    ("backbone_sha256", "wrong", "backbone_sha256"),
    ("split_hash", "wrong", "split_hash"),
    ("router_config", {}, "config"),
    ("queries_are_trained", False, "trained"),
    ("training", {}, "provenance"),
])
def test_router_provenance_rejected(inputs, field, value, match):
    _, model, manifest, bundle, options = inputs
    bundle[field] = value
    torch.save(bundle, options.router_checkpoint)
    with pytest.raises(ValueError, match=match):
        load_router_bundle(options.router_checkpoint, sha256_file(options.checkpoint),
                           manifest["sha256"], model, PruningConfig())


def test_checkpoint_contract_and_observer_validation(inputs):
    args, model, manifest, bundle, options = inputs
    checkpoint = torch.load(options.checkpoint, weights_only=True)
    checkpoint["config"]["use_cropr"] = True
    with pytest.raises(ValueError, match="QAT-only"):
        inherited_args(checkpoint, options)
    uncalibrated = small_fixture_model(args)
    with pytest.raises(ValueError, match="observers"):
        validate_qat_state(uncalibrated)
    from vitqat.quantization import Quantizer
    next(m for m in model.modules() if isinstance(m, Quantizer)).fake_quant.disable_fake_quant()
    with pytest.raises(ValueError, match="fake quantization"):
        validate_qat_state(model)


def test_split_mismatch_fails_before_outputs(inputs):
    from pathlib import Path
    _, _, _, _, options = inputs
    checkpoint = torch.load(options.checkpoint, weights_only=True)
    checkpoint["split_hash"] = "different"
    torch.save(checkpoint, options.checkpoint)
    with pytest.raises(ValueError, match="split"):
        run(options, model_factory=small_fixture_model)
    assert not Path(options.output).exists()


def test_missing_query_mapping_fails_before_evaluation(inputs):
    from pathlib import Path
    _, _, _, bundle, options = inputs
    del bundle["queries"]["1"]
    torch.save(bundle, options.router_checkpoint)
    with pytest.raises(ValueError, match="mapping"):
        run(options, model_factory=small_fixture_model)
    assert not Path(options.output).exists()
