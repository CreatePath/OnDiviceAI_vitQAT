import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

from vitqat.cli import parser
from vitqat.cropr_adapter import adapt_model, CrossAttention
from A3.inference_pruning import InferencePrunedViT, PruningConfig, QueryRouter
from A3.metrics import measure_vit
from vitqat.models import build_model
from vitqat.quantization import prepare_qat


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def settings(tmp_path, mode="qat"):
    return parser().parse_args([
        "--mode", mode, "--weight-bits", "4", "--activation-bits", "8",
        "--image-size", "224", "--workers", "0", "--device", "cpu",
        "--drop-path", "0", "--batch-size", "64", "--output", str(tmp_path / "training"),
    ])


def small_fixture_model(args):
    """Real 12-block adapter/QAT contract, narrow width for fast CPU tests."""
    base = VisionTransformer(img_size=args.image_size, patch_size=16, num_classes=101,
                             embed_dim=12, depth=12, num_heads=3, drop_path_rate=args.drop_path)
    model = adapt_model(base, args)
    if args.mode == "qat":
        prepare_qat(model, args.weight_bits, args.activation_bits, args.qat_scope)
    return model


def fixture_queries(dim=12):
    # Synthetic test tensors only; never exported as a trained research router.
    return {str(i): torch.randn(1, 1, dim) for i in range(1, 11)}


def test_scorer_selector_matches_upstream_per_image():
    query = torch.tensor([[[1., 0., 0., 0.]]])
    x = torch.tensor([[[0., 0., 0., 0.], [5., 1., 0., 0.], [1., 2., 0., 0.], [3., 3., 0., 0.]],
                      [[0., 0., 0., 0.], [1., 1., 0., 0.], [5., 2., 0., 0.], [3., 3., 0., 0.]]])
    upstream = CrossAttention(embed_dim=4, num_heads=1, num_queries=1, training=False)
    upstream.queries.data.copy_(query)
    router = QueryRouter(query, 2).eval()
    assert torch.equal(router.scores(x), upstream.forward_scorer(x))
    keep, dropped = router(x)
    assert torch.equal(keep[:, 0], x[:, 0])
    assert torch.equal(keep[0, 1], x[0, 1]) and torch.equal(keep[1, 1], x[1, 2])
    assert dropped.shape == (2, 2, 4)


@pytest.mark.parametrize("mode", ["fp32", "qat"])
def test_original_forward_equivalence_llf_and_metrics(tmp_path, mode):
    args = settings(tmp_path, mode)
    dense = small_fixture_model(args)
    images = torch.randn(2, 3, 224, 224)
    if mode == "qat":
        dense(images)  # Initialize synthetic fixture observers, not real task training.
    dense.eval()
    queries = fixture_queries()
    rng = torch.get_rng_state().clone()
    wrapper = InferencePrunedViT(dense, queries)
    assert torch.equal(rng, torch.get_rng_state())
    wrapper.enabled = False
    assert torch.equal(wrapper(images), dense(images))
    wrapper.enabled = True
    args.use_cropr, args.cropr_llf, args.cropr_pruning_rate = True, True, 16
    args.qat_scope = "backbone"
    upstream = small_fixture_model(args)
    mismatch = upstream.load_state_dict(dense.state_dict(), strict=False)
    assert not mismatch.unexpected_keys
    assert all(k.startswith("cropr.") for k in mismatch.missing_keys)
    for i, cropr in enumerate(upstream.cropr):
        cropr.cross_attn.queries.data.copy_(queries[str(i + 1)])
    upstream.eval()
    wrapper.capture_trace = True
    assert torch.allclose(wrapper(images), upstream(images), atol=1e-6, rtol=1e-5)
    trace = wrapper.last_trace
    assert trace["tokens_per_block"] == [197,180,164,148,132,116,100,84,68,52,36,197]
    assert torch.equal(trace["restored_ids"].sort(dim=1).values, torch.arange(197).expand(2, -1))
    assert trace["restored_ids"][:, 0].eq(0).all()
    assert sum(ids.shape[1] for ids in trace["removed_ids"]) == 161
    assert torch.equal(wrapper(images), upstream(images))
    wrapper.capture_trace = False
    wrapper(images)
    assert wrapper.last_trace is None
    metrics = measure_vit(wrapper, 224, "cpu", 0, 1, sample=images[:1])
    dense_metrics = measure_vit(dense, 224, "cpu", 0, 1)
    assert metrics["token_pruning_ratio"] == 161 / 196
    assert metrics["final_token_pruning_ratio"] == 0
    assert metrics["router_params"] == 120
    assert metrics["params"] == dense_metrics["params"] + 120
    assert metrics["inference_params"] == metrics["params"]
    ns = [197,180,164,148,132,116,100,84,68,52,36,197]
    d = 12
    expected_flops = 2 * 196 * d * (3 * 16 * 16) + 2 * d * 101
    expected_flops += sum(24 * n * d * d + 4 * n * n * d for n in ns)
    expected_flops += sum(2 * n * d for n in ns[:10])
    assert metrics["gflops"] == expected_flops / 1e9
    assert metrics["gflops"] < dense_metrics["gflops"]
    assert all(not p.requires_grad for p in wrapper.routers.parameters())
    with pytest.raises(RuntimeError, match="cannot train"):
        wrapper.train()(images)


def test_validation_no_random_or_invalid_schedule(tmp_path):
    model = small_fixture_model(settings(tmp_path, "fp32"))
    with pytest.raises(ValueError, match="mapping"):
        InferencePrunedViT(model, {})
    queries = fixture_queries()
    queries["1"] = torch.zeros(1, 1, 13)
    with pytest.raises(ValueError, match="dimension"):
        InferencePrunedViT(model, queries)
    with pytest.raises(ValueError, match="finite"):
        QueryRouter(torch.full((1, 1, 12), float("nan")), 16)
    with pytest.raises(ValueError, match="retain"):
        PruningConfig().schedule(12, 16)
    with pytest.raises(ValueError, match="LLF"):
        PruningConfig(llf=False).schedule(12, 196)


def test_documented_small_w4a8_forward(tmp_path):
    args = settings(tmp_path)
    args.model = "vit_small_patch16_224"
    model = build_model(args)
    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model(image)  # Synthetic observer initialization; not task training.
        model.eval()
        pruned = InferencePrunedViT(model, fixture_queries(384))
        pruned.capture_trace = True
        output = pruned(image)
    assert output.shape == (1, 101) and torch.isfinite(output).all()
    assert pruned.last_trace["tokens_per_block"] == [197,180,164,148,132,116,100,84,68,52,36,197]
    assert sum(p.numel() for p in pruned.routers.parameters()) == 3840


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_pruned_qat_cuda(tmp_path):
    model = small_fixture_model(settings(tmp_path)).cuda()
    image = torch.randn(1, 3, 224, 224, device="cuda")
    model(image)
    model.eval()
    wrapper = InferencePrunedViT(model, fixture_queries()).cuda()
    measured = measure_vit(wrapper, 224, "cuda", 0, 1, sample=image)
    assert measured["token_pruning_ratio"] == 161 / 196
    assert measured["latency_ms_median"] > 0
