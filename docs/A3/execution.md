# 실행 방법과 query 인수 형식

## 구현 범위

`A3.inference_pruning`과 `A3.eval_pruning`을 구현했다. 선행 QAT-only checkpoint의 학습 설정을 복원하고, 학습된 query를 별도로 로드해 dense/pruned를 한 번에 평가한다. 모델/bit/epochs/LR 등의 재지정 옵션은 두지 않았다. 학습이나 query 초기화·학습을 실행하지 않는다.

현재 loader는 저장소 `vitqat.train`이 저장한 QAT-only checkpoint와 Food-101 데이터 경로를 지원한다. 다른 구현, pretrained/custom pooling/custom preprocessing 모델은 해당 모델을 그대로 재현하는 loader adapter가 필요하다. 지원되지 않는 조건을 기본 Small/W4A8로 대체하지 않는다.

## 실행

저장소의 `environment.yml` 환경을 활성화한다. 아래 경로는 인수 파일의 예시다. **실제 학습된 query 파일이 있어야 한다.**

```powershell
conda activate vitqat
python -m A3.eval_pruning --checkpoint A3/checkpoints/qat-only/seed42/best.pt --router-checkpoint A3/checkpoints/qat-only/seed42/cropr_queries.pt --output A3/runs/seed42/eval-pair --device cuda
```

기본 preset은 `cropr-r16-first17-llf`다. R=16, 첫 모듈 17개, 이후 각 16개 제거, 마지막 두 block 뒤 pruning 생략, LLF on, FP32 query router를 사용한다. 예상 12-block/224/patch16에서 1~10번 뒤 제거, 최대 TPR 161/196이다. 실제 depth/patch 수는 선행 모델에서 읽으며 잘못된 제거량이나 query 구조는 거부한다.

| 옵션 | 기본값 / 의미 |
|---|---|
| `--checkpoint` | 필수, 선행 QAT-only best checkpoint |
| `--router-checkpoint` | 필수, 아래 형식의 학습된 query bundle |
| `--output` | 필수, 새 평가 폴더. 기존 결과 덮어쓰기 거부 |
| `--evaluation-id` | `S-QAT-P`, 원래 학습 experiment와 별도 저장 |
| `--pruning-schedule` | `cropr-r16-first17-llf`, 현재 단일 preset |
| `--data-root` | 생략 시 선행 config. 위치만 변경하며 split은 유지 |
| `--device` | 가용 시 cuda, 아니면 cpu |
| `--workers`, `--threads` | 생략 시 선행 config (`threads`가 없으면 0) |
| `--eval-batch-size` | 생략 시 선행 batch size. 지정 시 양쪽 평가에 동일 적용 |
| `--latency-warmup`, `--latency-repeats` | 10 / 100 |

Bit 수도 선행 실험의 실제 값을 상속한다. Observer와 fake quantization 상태가 유효해야 한다. 새 calibration이나 fake quantization 해제는 하지 않는다. 학습 LR/epoch 설정은 기록만 하며 추가 학습에 사용하지 않는다.

## Router checkpoint 계약

`torch.save()` dictionary이며 `weights_only=True`로 읽는다. Queries는 **해당 backbone에 맞춰 이미 학습한 tensor**여야 한다. 아래는 구성 예시이며 `learned_queries`는 별도 학습 산출물로 준비해야 한다. Query를 생성하는 코드가 아니다.

```python
import torch
from A3.eval_pruning import sha256_file
from A3.inference_pruning import PruningConfig

# learned_queries: {"1": tensor[1,1,D], ..., "10": tensor[1,1,D]}
# actual_training: 아래 항목을 포함한 실제 학습 이력
bundle = {
    "format_version": 1,
    "queries_are_trained": True,
    "backbone_sha256": sha256_file("A3/checkpoints/qat-only/seed42/best.pt"),
    "split_hash": upstream_split_hash,
    "router_config": PruningConfig().describe(depth=12, patch_tokens=196),
    "queries": learned_queries,
    "training": actual_training,
}
torch.save(bundle, "A3/checkpoints/qat-only/seed42/cropr_queries.pt")
```

`training` 필수 항목:

| Key | 형식 / 내용 |
|---|---|
| `method` | 비어 있지 않은 문자열, query 학습 절차 |
| `data` | 비어 있지 않은 문자열, 학습 dataset/split |
| `backbone_frozen` | bool, query 학습 중 backbone 고정 여부 |
| `observers_frozen` | bool, query 학습 중 observer 고정 여부 |
| `pruning_during_training` | bool, query 학습 중 pruning 노출 여부 |

실제 epochs/LR/loss/코드 commit 등의 추가 필드도 JSON 값으로 기록한다. 학습 완료 표시와 provenance는 제공자의 선언이며 evaluator가 학습의 진위를 증명하는 것은 아니다. 무작위 query에 학습 완료 표시를 해서 연구에 사용하지 않는다.

다른 depth에서는 모듈 수가 `depth-2`이며 key는 1부터 모듈 수까지의 문자열이다. Tensor D는 실제 backbone embedding 차원과 같아야 한다. 유효한 값, key, config, backbone/split hash를 모두 검사한다. 기존 Cropr 전체 checkpoint는 query bundle로 바로 읽지 않는다. 학습된 query를 추출하고 동일 backbone과의 관계를 확인해 위 형식으로 포장한다.

## 산출물

`evaluation_config.json`은 원래 학습 config와 실제 평가 config, pruning 설정, 두 입력 SHA256, query provenance를 저장한다. `split.json`, `dense_metrics.json`, `pruned_metrics.json`, `comparison.json/csv`도 저장한다.

같은 test split에서 정확도를 측정한다. Latency는 seed로 고른 같은 validation image 1장을 사용하며 subset/dataset index를 기록한다. 전체 forward에 scoring/sorting/gather/LLF 비용을 포함하고 FLOPs에는 query dot product를 포함한다. Backbone/quantizer/query는 평가 전후 byte digest로 불변을 확인한다.

Query parameter 수를 별도 기록한다. Optimizer 포함 checkpoint 파일 크기를 추론 모델 크기로 보고하지 않는다. Fake-quant latency는 실제 저비트 kernel latency가 아니다.

## 검증

```powershell
python -m pytest tests A3/tests -q
python -m A3.eval_pruning --help
```

생성 이미지·임의 query는 테스트 fixture이며 실제 Food-101 성능 결과가 아니다. 연구 평가는 선행 QAT 모델과 학습된 query를 인수한 뒤 수행한다.

2026-10-26 로컬 CPU 검증: `A3/tests`는 **14 passed / 1 skipped**, 기존 `tests`는 **59 passed / 6 skipped**였다. CUDA가 없어 GPU 테스트는 건너뛰었다. A3 CLI 도움말, 원본 Cropr와의 forward 동등성, 실제 ViT-S/16 W4A8 구조의 합성 입력 forward도 확인했다.

검증 환경은 별도 임시 venv의 Python 3.13.5, torch 2.12.0+cpu, torchvision 0.27.0+cpu, timm 1.0.3이었다. 저장소가 지정한 Python 3.10 / torch 2.3.0+cu121 환경과의 차이를 기록하며, CUDA 환경 및 실제 task 성능은 이번 검증에 포함하지 않는다. 기존 `vitqat/`, `cropr/`, `tests/`의 tracked 파일에는 변경이 없다.
