import copy

import pytest
import torch

from A3.pruning import attach_pruning, PrunedOutput
from A3.tests.test_inference_pruning import settings, small_fixture_model, fixture_queries


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def test_inference_api_dense_train_pruned_eval(tmp_path):
    backbone = small_fixture_model(settings(tmp_path, "fp32"))
    keys = set(backbone.state_dict())
    params = {k: v.clone() for k, v in backbone.named_parameters()}
    wrapped = attach_pruning(backbone, queries=fixture_queries())
    assert wrapped.training and backbone.training
    x = torch.randn(2, 3, 224, 224)
    assert torch.equal(wrapped(x), backbone(x))
    assert set(backbone.state_dict()) == keys
    assert all(torch.equal(p, params[k]) for k, p in backbone.named_parameters())
    assert all(not p.requires_grad for p in wrapped.routers.parameters())
    wrapped.eval()
    wrapped.capture_trace = True
    assert wrapped(x).shape == (2, 101)
    assert wrapped.last_trace["tokens_per_block"][-2:] == [36, 197]
    wrapped.enabled = False
    assert torch.equal(wrapped(x), backbone(x))
    with pytest.raises(ValueError, match="already attached"):
        attach_pruning(wrapped, queries=fixture_queries())


def test_train_api_gradients_export_reload_and_loss(tmp_path):
    backbone = small_fixture_model(settings(tmp_path, "fp32"))
    wrapped = attach_pruning(backbone, apply_during="train_and_inference")
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-3)
    x, y = torch.randn(2, 3, 224, 224), torch.tensor([1, 2])
    out = wrapped(x)
    assert isinstance(out, PrunedOutput) and len(out.auxiliary_logits) == 10
    criterion = torch.nn.CrossEntropyLoss()
    criterion(out.logits, y).backward(retain_graph=True)
    assert wrapped.cropr[0].cross_attn.queries.grad is None
    optimizer.zero_grad(set_to_none=True)
    out.auxiliary_loss(y, criterion).backward()
    assert backbone.patch_embed.proj.weight.grad is None  # stop-gradient
    assert wrapped.cropr[0].cross_attn.queries.grad.abs().sum() > 0
    optimizer.step()
    state = copy.deepcopy(wrapped.state_dict())
    wrapped.eval()
    expected = wrapped(x)
    inference = attach_pruning(backbone, queries=wrapped.export_queries()).eval()
    assert torch.allclose(inference(x), expected, atol=1e-6)
    restored = attach_pruning(small_fixture_model(settings(tmp_path, "fp32")),
                             apply_during="train_and_inference")
    restored.load_state_dict(state, strict=True)
    assert torch.equal(restored.eval()(x), expected)


def test_errors_and_soft_target_loss(tmp_path):
    backbone = small_fixture_model(settings(tmp_path, "fp32"))
    with pytest.raises(ValueError, match="learned queries"):
        attach_pruning(backbone)
    with pytest.raises(ValueError, match="apply_during"):
        attach_pruning(backbone, apply_during="bad")
    from timm.loss import SoftTargetCrossEntropy
    model = attach_pruning(backbone, apply_during="train_and_inference")
    target = torch.nn.functional.one_hot(torch.tensor([1, 2]), 101).float()
    out = model(torch.randn(2, 3, 224, 224))
    loss = SoftTargetCrossEntropy()(out.logits, target) + out.auxiliary_loss(target, SoftTargetCrossEntropy())
    loss.backward()
    assert torch.isfinite(loss)


def test_qat_train_api_optimizer_and_eval_state(tmp_path):
    from vitqat.quantization import Quantizer
    from A3.eval_pruning import state_digest
    backbone = small_fixture_model(settings(tmp_path, "qat"))
    model = attach_pruning(backbone, apply_during="train_and_inference")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    query = model.cropr[0].cross_attn.queries
    before = query.detach().clone()
    out = model(torch.randn(2, 3, 224, 224))
    y = torch.tensor([1, 2])
    ce = torch.nn.CrossEntropyLoss()
    (ce(out.logits, y) + out.auxiliary_loss(y, ce)).backward()
    assert backbone.patch_embed.proj.weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(query, before)
    model.eval()
    for module in backbone.modules():
        if isinstance(module, Quantizer):
            module.fake_quant.disable_observer()
    original = state_digest(model)
    with torch.inference_mode():
        logits = model(torch.randn(2, 3, 224, 224))
    assert logits.shape == (2, 101) and torch.isfinite(logits).all()
    assert state_digest(model) == original
