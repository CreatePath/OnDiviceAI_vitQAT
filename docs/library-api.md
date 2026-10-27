# Token pruning 라이브러리 사용법

## 1. 이 라이브러리로 할 수 있는 일

- `A3/pruning/`에서 Cropr token pruning을 가져와 기존 모델에 연결한다.
- 학습과 추론에서 pruning을 적용할 시점을 선택한다.
- 기존 `vitqat/`와 `cropr/` 코드는 수정하지 않는다. 기존 모델 객체를 wrapper로 감싼다.
- A3 기본 실험은 **A2 QAT-only 모델에 추론 시에만 pruning을 적용**한다.
- A3 평가 프로그램도 이 라이브러리를 사용한다. 평가 명령과 checkpoint 형식은 [A3 실행 방법](A3/execution.md)을 참고한다.

```python
from A3.pruning import attach_pruning, PruningConfig
```

## 2. 두 가지 사용 방식

| 설정 | 학습 시 | `eval()` 시 | Query 준비 |
|---|---|---|---|
| `apply_during="inference"` | 전체 토큰 사용 | pruning 적용 | 학습된 query 필수 |
| `apply_during="train_and_inference"` | pruning과 보조 head 적용 | pruning 적용 | 생략하면 학습할 query를 새로 초기화 |

- `model.enabled = False`로 설정하면 pruning을 끌 수 있다.
- `eval()`은 검증과 추론에 모두 사용하므로 inference 모드에서 validation에도 pruning이 적용된다.
- 이 라이브러리는 데이터 준비, 학습 반복, checkpoint 저장을 자동 실행하지 않는다.

## 3. A3: 추론할 때만 pruning

1. A2 모델을 만들고 학습된 QAT checkpoint를 불러온다.
2. 해당 모델에 맞춰 학습된 Cropr query를 준비한다.
3. 모델에 pruning 기능을 연결한다.
4. `eval()`로 전환한 뒤 추론한다.

```python
import torch
from A3.pruning import attach_pruning, PruningConfig

# backbone: A2 체크포인트를 로드한 모델
# queries: block 번호별 학습된 query tensor 딕셔너리
# images: 전처리된 입력
model = attach_pruning(
    backbone,
    queries=queries,
    config=PruningConfig(pruning_rate=16, llf=True),
    apply_during="inference",
)

model.eval()
with torch.inference_mode():
    logits = model(images)  # 추론에서 pruning 적용
```

- `train()`에서는 기존 backbone을 그대로 실행하고 query는 고정된다.
- API에 전달하는 query는 tensor다. 파일 출처와 학습 여부를 직접 확인해야 한다.
- A3 평가 CLI는 별도 query bundle의 backbone/split hash, 설정, 학습 이력을 검사한다. 해당 모델과 맞는 query가 없으면 실행할 수 없다.
- 모델을 다시 학습하면 feature 표현이 바뀔 수 있다. 그 후 기존 query가 여전히 맞는지 확인해야 한다.

## 4. 학습과 추론 모두 pruning

- 이 모드는 다른 실험에서 재사용하도록 제공한다. A3의 기본 실험은 여전히 추론 전용이다.
- 학습 시 원본 Cropr의 query, aggregator, 보조 분류 head를 사용한다.
- 학습 시 반환값은 `PrunedOutput`이다. `logits`와 `auxiliary_logits`를 제공한다.
- **보조 loss를 최종 loss에 더해야 query가 학습된다.** 기존 `vitqat.training.train_epoch`는 이 반환형을 자동 처리하지 않으므로 호출하는 쪽에서 학습 step을 작성해야 한다.
- Optimizer는 pruning을 연결한 **뒤** 생성해야 query와 보조 head가 포함된다.

```python
import torch
from A3.pruning import attach_pruning

# backbone, images, target, criterion, upstream_lr는 호출하는 쪽에서 준비한다.
model = attach_pruning(backbone, apply_during="train_and_inference")
optimizer = torch.optim.AdamW(model.parameters(), lr=upstream_lr)

model.train()
out = model(images)
main_loss = criterion(out.logits, target)
aux_loss = out.auxiliary_loss(target, criterion)  # 각 보조 loss의 합
loss = main_loss + 1.0 * aux_loss                 # 계수는 실험 설정에 맞춘다
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()

model.eval()
with torch.inference_mode():
    logits = model(images)  # 보조 head 없이 pruning 추론
```

- Mixup/CutMix를 사용할 때는 main과 auxiliary에 같은 soft target 및 적절한 loss 함수를 사용한다.
- 보조 loss의 gradient는 stop-gradient를 통해 backbone으로 전달되지 않는다. Main loss는 backbone을 학습한다.
- Query만 학습하려면 호출자가 backbone parameter와 QAT observer를 별도로 고정해야 한다.
- 마지막 block DropPath 등 backbone 설정은 호출자가 관리한다. 논문의 LLF 학습 조건을 따를 때는 마지막 block DropPath를 꺼야 한다.

## 5. Query 저장과 재사용

- `train_and_inference` wrapper의 `export_queries()`는 query tensor만 반환한다.
- 평가 CLI에 사용할 파일에는 tensor 외에 backbone/split hash와 실제 학습 이력을 기록해야 한다. [Query bundle 계약](A3/execution.md)을 따른다.
- Query 학습 완료 표시는 제공자의 이력 선언이며 파일 검사만으로 학습의 진위를 증명할 수 없다.
- 학습 재개 시에는 wrapper 전체 `state_dict()`, optimizer/RNG 상태와 설정을 함께 저장한다. 같은 wrapper 구조에 strict load한다.
- Wrapper의 `backbone`은 원래 모델 객체를 공유한다. 여러 wrapper를 번갈아 사용할 때는 `train()`/`eval()` 상태를 명시한다.

```python
queries = model.export_queries()  # train_and_inference wrapper에서 호출
inference_model = attach_pruning(model.backbone, queries=queries).eval()
```

## 6. 기본 pruning 설정

| 항목 | 기본값 |
|---|---|
| 제거 개수 | `pruning_rate=16`, 첫 단계는 17개 |
| 위치 | LLF 사용 시 마지막 두 블록을 제외한 각 블록 뒤 |
| CLS token | 항상 유지 |
| LLF | 켜짐. 제거 토큰을 마지막 블록 앞에 복원 |
| Query scoring | 학습된 query와 token feature의 내적, Top-K 선택 |

- 12블록, 224 입력, patch16에서는 1~10번 블록 뒤에 총 161개 patch를 제거한다. 최대 TPR은 161/196, 약 82.14%다.
- 모델의 depth와 patch 수가 다르면 위치와 실제 제거율도 달라진다. 토큰을 모두 제거하는 설정은 오류로 처리한다.
- `PruningConfig`를 전달하면 config 값이 우선한다. 현재 LLF를 끄는 설정은 지원하지 않는다.
- A3의 모델·QAT·데이터·학습 하이퍼파라미터는 선행 checkpoint 설정을 따른다.

## 7. 지원 범위와 검증

- 지원 모델: 이 저장소의 dense Cropr adapter ViT. `forward_embed`, `forward_head`, `blocks`, `norm` 등의 인터페이스가 필요하다.
- 지원 정밀도: FP32 또는 기존 fake-quant FP32 backbone. 임의의 timm 모델과 AMP 전체를 지원한다고 보장하지 않는다.
- 이미 pruning이 적용된 모델의 중복 연결은 거부한다. 기존 `A3.inference_pruning` import는 호환용으로 유지한다.
- CPU 테스트: **77개 통과, 7개 건너뜀**. GPU가 없어 CUDA 테스트는 실행하지 않았다.
- 테스트 범위: 학습/추론 분기, 보조 loss gradient, query 업데이트, soft target, QAT forward, 저장·복원과 query export.
- 실제 Food-101 학습 수렴, A2 checkpoint와 학습된 query의 결합 성능, GPU 속도는 아직 측정하지 않았다.
