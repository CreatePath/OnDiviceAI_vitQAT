# CLS attention Top-K 토큰 pruning 구현 기록

작성일: 2026-10-27

## 목적과 범위

ViT 학습과 추론에서 CLS 토큰이 각 패치에 주는 self-attention을 기준으로 중요도가 낮은 토큰을 제거한다. Cropr의 pruning 위치, 제거 개수, 마지막 블록에서의 토큰 복원(LLF)을 사용한다. 토큰을 고르는 방식은 별도 query를 학습하는 Cropr scorer가 아니다.

기존 FP32 또는 fake-quant QAT backbone에 `A3.pruning.attach_pruning`을 연결한다. 프로젝트의 dense Cropr adapter ViT 인터페이스와 CLS 토큰 하나가 필요하며, 임의의 ViT 모델과 실제 정수 연산 커널을 지원한다고 가정하지 않는다.

## 토큰 선택과 스케줄

1. 각 pruning 대상 블록의 self-attention에서 CLS 행을 가져온다.
2. 모든 attention head의 값을 평균하여 패치별 점수를 만든다. 학습 중에는 attention dropout 이후, QAT에서는 그 뒤 fake quantization까지 적용된 값을 사용한다.
3. CLS를 항상 유지하고 점수가 높은 패치부터 남긴다. 이때 이미지마다 독립적으로 순위를 계산한다. 동점은 `argsort(..., stable=False)`의 순서에 따른다.
4. 제거된 토큰은 보관했다가 마지막 transformer 블록 입력 전에 남은 토큰과 합친다(LLF).

기본 `pruning_rate=16`에서는 첫 단계에서 패치 17개를 제거하고 이후 각 단계에서 16개를 제거한다. 마지막 두 블록 뒤에는 제거하지 않는다. 12블록, 패치 196개 모델에서 블록 1~10 뒤에 총 161개를 제거한다. 각 블록 입력 토큰 수는 CLS 포함 `[197, 180, 164, 148, 132, 116, 100, 84, 68, 52, 36, 197]`이다. 실제 depth와 patch 수가 다르면 스케줄도 달라지며 최소 한 개의 패치는 유지해야 한다.

## 라이브러리 사용

| `apply_during` | `train()` | `eval()` | 반환값 |
|---|---|---|---|
| `inference` (기본) | dense | CLS attention Top-K | logits |
| `train` | CLS attention Top-K | dense | logits |
| `train_and_inference` | CLS attention Top-K | CLS attention Top-K | logits |
| `cropr_train_and_inference` | 학습형 query 기반 Cropr | query 기반 Cropr | 학습 시 `PrunedOutput`, 평가 시 logits |

```python
import torch
from A3.pruning import attach_pruning, PruningConfig

# backbone: 필요한 체크포인트를 로드하고, QAT 사용 시 QAT 준비를 마친 모델
model = attach_pruning(
    backbone,
    config=PruningConfig(pruning_rate=16, llf=True),
    apply_during="train_and_inference",
)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model.train()
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, targets)
loss.backward()
optimizer.step()

model.eval()
with torch.inference_mode():
    logits = model(images)
```

CLS attention 모드에는 query와 보조 loss가 필요 없고 router의 학습 파라미터도 없다. `model.enabled = False`로 pruning을 끌 수 있다. Wrapper는 입력 backbone 객체를 공유한다. 기존 `train_and_inference`가 Cropr query를 사용하던 호출은 `cropr_train_and_inference`로 변경해야 한다. 기존 backbone 체크포인트는 연결 전에 로드하고, wrapper 전체 체크포인트를 재개할 때는 같은 모드와 설정으로 wrapper를 만든 뒤 모델과 optimizer 상태를 복원한다.

Top-K의 이산적인 선택 자체에는 gradient가 흐르지 않는다. 선택된 토큰과 LLF로 복원된 토큰을 통과하는 backbone 연산에는 gradient가 흐른다. 학습 중 dropout이나 QAT quantizer의 동작이 평가 때와 다르면 같은 이미지에서도 선택 토큰이 달라질 수 있다.

## 검증 결과와 남은 작업

2026-10-27 CPU 합성 fixture 검증:

- `python -m pytest A3/tests -q`: **16개 통과, 1개 건너뜀**. FP32와 QAT에서 학습 전용 모드가 학습 중 pruning하고 평가 시 dense로 실행되는지 확인했다. CUDA가 없어 GPU 테스트는 건너뛰었다.
- `python -m pytest A3/tests/test_pruning_api.py -q`: **6개 통과**. FP32 및 fake-quant QAT 학습에서 실제 attention 기반 제거, 토큰 수, backbone gradient와 optimizer 업데이트, 평가 결과, wrapper 저장·복원, pruning 비활성화를 확인했다.
- 학습 모드 추가 이전의 전체 테스트 `python -m pytest tests A3/tests -q`: **71개 통과, 7개 건너뜀**. 이는 새 학습 모드 추가 이후 전체 테스트 결과가 아니다.

실제 Food-101 데이터와 A2 체크포인트를 사용한 학습 수렴, 정확도 변화, GPU latency 및 실제 정수 연산 속도는 측정하지 않았다. 합성 테스트 통과를 모델 성능이나 실제 가속의 근거로 사용하지 않는다. QAT-only checkpoint의 dense/pruned 추론 비교 명령은 [A3 실행 방법](execution.md)을 참조한다.

구현은 [inference.py](../../A3/pruning/inference.py), 모드 선택은 [model.py](../../A3/pruning/model.py), 상세 사용법은 [라이브러리 API](../library-api.md)를 참조한다.
