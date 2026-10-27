# Token pruning 라이브러리 사용법

구현 범위와 검증 결과는 [CLS attention pruning 구현 기록](A3/cls-attention-pruning.md)을 참조한다.

A3 추론은 기존 ViT의 CLS-to-patch self-attention 확률을 head별로 평균하여 토큰 순위를 매긴다. 별도 학습 query는 필요 없다. QAT에서는 forward에 사용되는 fake-quant attention 확률을 사용한다.

```python
import torch
from A3.pruning import attach_pruning, PruningConfig

model = attach_pruning(backbone, config=PruningConfig(pruning_rate=16, llf=True))
model.eval()
with torch.inference_mode():
    logits = model(images)
```

기본 추론 wrapper는 `train()`에서 dense backbone을 실행하고 `eval()`에서 pruning한다. `model.enabled = False`로 pruning을 끌 수 있다. Backbone 객체는 공유한다.

`apply_during="train"`을 지정하면 `train()`에서만 CLS attention pruning을 적용하고 `eval()`에서는 dense backbone을 사용한다. 학습과 평가 양쪽에 적용하려면 `apply_during="train_and_inference"`를 사용한다.

위치와 개수는 Cropr 스케줄을 따른다. 첫 블록 뒤에 패치 17개, 이후 pruning 단계마다 16개를 제거한다. CLS는 유지하고 제거된 토큰은 마지막 블록 전에 복원한다(LLF). 12블록·196패치 모델에서는 1~10번 블록 뒤에 총 161개를 제거한다. CLS attention Top-K는 Cropr의 학습형 query scorer와 다르다.

학습과 추론에 같은 CLS attention Top-K를 적용하려면 다음과 같이 사용한다. 반환값은 일반 logits 텐서이며 추가 보조 loss나 query가 필요 없다.

```python
model = attach_pruning(backbone, apply_during="train_and_inference")
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

학습 시 각 블록의 attention dropout 이후 확률을 점수에 사용한다. QAT라면 그 확률의 fake quantization 이후 값을 사용한다. Top-K 선택 자체는 미분되지 않지만 선택된 토큰 및 LLF로 복원된 토큰을 통한 backbone gradient는 전달된다. Dropout이나 양자화로 학습과 평가 시 선택 토큰이 달라질 수 있다.

기존 backbone 체크포인트는 pruning을 연결하기 전에 backbone에 로드한다. 학습 재개를 위해 wrapper 전체의 `state_dict()`를 저장했다면 동일한 모드와 설정으로 wrapper를 다시 연결한 다음 로드하고 optimizer 상태도 복원한다. QAT를 쓸 때는 backbone에 QAT 모듈을 준비한 후 wrapper를 연결한다.

기존 학습형 query와 보조 head를 사용하는 Cropr 경로는 `apply_during="cropr_train_and_inference"`로 지정한다. 이전의 `train_and_inference` 호출이 Cropr를 사용했다면 이 새 이름으로 변경해야 한다. `export_queries()`와 `PrunedOutput`은 이 Cropr 경로 전용이다.
