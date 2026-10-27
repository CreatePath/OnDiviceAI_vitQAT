> **Current implementation (2026-10-27):** The following is the previous learned-query design and paper notes. A3 inference now selects tokens using CLS self-attention Top-K, retaining the Cropr removal schedule and LLF. No query checkpoint is required. See [execution](execution.md).

# 실험 설계

## 1. 담당 범위와 연구 질문

다른 담당자가 FP32 원본과 QAT-only 모델을 학습한다. 이 작업에서는 **전달받은 QAT-only backbone을 고정하고 추론 시 Cropr를 적용**한다. 동일한 backbone의 pruning off/on에서 Top-1 손실, 연산량, latency를 비교한다.

논문 Cropr는 학습형 query를 요구한다. Query를 별도 학습해서 제공받으면 그 이력도 결과에 명시한다. QAT-only checkpoint만으로 학습된 Cropr router를 구성할 수 없으며, query 확보 전에는 실제 pruning 평가를 완료할 수 없다. 상세 근거와 미결정 사항은 [pruning 파라미터](pruning-parameters.md) 5절에 정리했다.

## 2. 비교군

| ID | 입력 | 이 작업의 실행 | 역할 |
|---|---|---|---|
| S-FP32-D | 다른 담당자 FP32 결과 | 전달 결과 참고 | 프로젝트 전체의 FP32 기준선 |
| S-QAT-D | 다른 담당자 QAT-only best checkpoint | dense 추론 재평가 | 주 기준선 |
| S-QAT-P | 동일 QAT backbone + 별도 학습된 Cropr queries | pruned 추론 평가 | 주 실험군 |

Backbone parameter와 quantizer state는 D/P 사이에 동일해야 한다. P에는 학습된 query가 추가되므로 전체 모델 state가 완전히 같다는 표현은 사용하지 않는다. Query의 task 학습 효과가 포함되므로 이 실험을 학습 없는 heuristic pruning과 동일한 조건으로 해석하지 않는다.

`--use-cropr`를 켜 새로 backbone을 학습하는 기존 경로는 이번 실행 범위에 포함하지 않는다. 학습형 query를 무작위로 붙이거나 CLS embedding으로 대체하지 않는다. FP32 pruning과 joint QAT+Cropr 학습도 현재 담당 범위 밖이다.

## 3. 전달받을 조건과 자료

### 선행 실험 설정을 그대로 따르는 원칙

본 실험은 앞선 **2번 ViT QAT-only 실험의 확장**이다. 아래 조건은 선행 실험의 최종 config/checkpoint를 기준으로 상속하며, Cropr 논문의 원래 태스크와 다르다는 이유로 변경하지 않는다.

| 조건 | 이번 실험의 적용 원칙 |
|---|---|
| 모델 구조, depth, embedding 차원, classifier, pooling | 선행 QAT-only 모델 그대로 사용 |
| Dataset, class mapping, train/val/test split | 선행 실험과 동일하게 사용 |
| 입력 크기, patch 크기, 평가 전처리 | 선행 실험과 동일하게 사용 |
| Scratch/pretrained, finetuning 방식, 학습 하이퍼파라미터 | 선행 실험의 실제 값을 기록, 재학습하지 않음 |
| QAT bit 수, quantizer 구현과 학습된 observer state | 선행 checkpoint 그대로 사용, 평가 중 갱신하지 않음 |
| 정확도 평가 방법과 측정 환경 | 선행 조건을 따르고 dense/pruned를 같은 환경에서 비교 |
| 이번 실험에서 추가할 조건 | Cropr scorer/queries, Top-K, pruning 위치·제거량, LLF |

이 문서의 숫자 예시는 선행 실험 설정이 확정되면 실제 값으로 갱신한다. 선행 모델의 depth/patch 수가 예상과 다르면 모델을 예시에 맞춰 바꾸지 않는다. 상속한 구조에서 pruning schedule의 유효성, query 차원, token trace와 TPR을 다시 계산해 기록한다. 측정 장치를 동일하게 사용할 수 없으면 차이를 기록하고 같은 장치에서 dense/pruned를 재측정한다.

예상 조건은 Small/16, 입력 224, 101 classes, CLS pooling, scratch W4A8 QAT이다. Scratch/pretrained 선택과 full finetuning/head-only/LoRA 여부는 학습 담당자가 확정한 실제 config로 기록한다. 이 작업에서 모델을 재초기화하거나 학습 방식을 바꾸지 않는다.

| 학습 조건 | 기존 코드 예상값, 인수 시 실제 값 확인 |
|---|---|
| Epochs / warmup | 300 / 20 |
| Optimizer / LR | AdamW / 5e-4 |
| LR decay / min LR | linear warmup 후 cosine / 1e-6 |
| Batch / accumulation | 128 / 1 |
| Weight decay / label smoothing | 0.05 / 0.1 |
| Drop path / clip-grad | 0.1 / norm 1.0 |
| Mixup / CutMix / RandomErasing | 0 / 0 / 0 |
| Crop minimum scale | 0.08 |
| Seed / val fraction | 42 / 0.1 |
| QAT scope / freeze epoch | all / -1, 평가 중 observer 고정 |
| 학습 pruning | 없음 |
| AMP / TF32 | off / off |

이 표는 새 학습 지시가 아니라 인수 메타데이터다. 기본값 대신 실제 전달값을 재현에 사용한다. 모델 구조나 QAT 구현이 다르면 로딩 호환성을 먼저 확인한다.

필수 인수 자료:

- QAT-only `best.pt`: backbone과 quantizer/observer state, checkpoint SHA256.
- `config.json`, 학습 코드 commit 및 라이브러리 버전.
- `split.json`, class mapping, 평가 전처리.
- Dense `metrics.json/csv`와 best validation epoch 선택 근거.
- Cropr query checkpoint: block 1~10별 query, shape, SHA256, 학습에 사용한 backbone SHA256과 학습 조건.

Query 학습 이력에는 사용 데이터/split, backbone·observer 고정 여부, 학습 시 pruning 여부를 남긴다. 다른 backbone의 query는 같은 차원이라는 이유만으로 호환된다고 가정하지 않는다.

## 4. 추론 pruning 설정

세부 값과 논문 근거는 [파라미터 문서](pruning-parameters.md)를 따른다.

| 항목 | 값 |
|---|---|
| Scorer | learned-query dot product, 모듈마다 query 1개, head 1개 |
| Projection / scorer LN | 없음 |
| Router 정밀도 | FP32 제안, 논문 외 프로젝트 선택으로 기록 |
| 위치 | 1~10번 block 뒤 |
| 제거 개수 | `[17,16,16,16,16,16,16,16,16,16]` |
| CLS | 항상 보존 |
| LLF | on, 마지막 block 앞 복원 |
| Token trace | `[197,180,164,148,132,116,100,84,68,52,36,197]` |
| 최대 TPR / 최종 TPR | 161/196 = 82.14% / 0% |
| Schedule ID | `cropr-r16-first17-llf` |

Pooling은 선행 QAT-only 모델의 실제 설정을 그대로 유지한다(CLS는 현재 예상값). 모델·데이터·QAT 조건은 선행 실험에서 상속하며, 논문의 avg pooling이나 ImageNet 조건에 맞추기 위해 변경하지 않는다. Router 정밀도는 추가 pruning 모듈에만 해당하는 별도 설정이다.

## 5. 데이터와 평가 절차

예상 Food-101 split은 공식 train 75,750장 중 train 68,175 / validation 7,575, 공식 test 25,250장이다. 전달받은 split hash와 class 순서를 검증한다. 평가 전처리는 전달 설정을 재사용하며 기본은 Resize 256, CenterCrop 224, ImageNet normalization이다.

1. 전달받은 QAT best checkpoint와 config를 strict load한다. 같은 validation 기준으로 선택된 checkpoint를 그대로 사용한다.
2. Dense 평가에서 전달된 기준 정확도를 재현한다. 차이가 나면 전처리, split, class mapping, QAT 구현과 observer state를 확인한다.
3. 별도로 학습된 queries의 출처와 block mapping을 검증하고 router를 연결한다.
4. Backbone/observer를 갱신하지 않고 동일 test에서 pruned 평가한다. Test calibration과 optimizer step을 수행하지 않는다.
5. 같은 장치·입력·측정 조건에서 dense/pruned 지표를 기록한다.

Query 확보 방식이 정해지기 전에는 3번 이후를 실행하지 않는다. 별도 query 학습을 허용한다면 train split만 사용하고 validation으로 설정을 선택한다. Test로 query, R, 위치, checkpoint를 선택하지 않는다. 추가 seed checkpoint가 제공되면 seed별로 같은 backbone의 D/P를 짝지어 반복한다.

## 6. 실행 상태

`A3.inference_pruning`에 router/Top-K/LLF를, `A3.eval_pruning`에 별도 query 인수와 paired 평가를 구현했다. 기존 학습 CLI에 `--use-cropr`를 켜는 방식으로 대체하지 않는다. Dense checkpoint는 strict load하고 학습된 queries를 별도 연결한다.

실제 명령과 query bundle 계약은 [실행 방법](execution.md)에 정리했다. 신규 CLI는 원래 config를 checkpoint에서 복원하므로 학습 옵션을 다시 입력하지 않는다. FP32/QAT 학습 명령은 이 담당 문서에 포함하지 않는다. 실제 평가에는 학습된 query가 필요하다.

## 7. 결과 기록과 해석

| 조건 | Backbone SHA256 | Router SHA256 | Test Top-1 (%) | 최대/최종 TPR | GFLOPs | median/p95 latency |
|---|---|---|---|---|---|---|
| S-QAT-D | 전달 후 기록 | 없음 | 미측정 | 0 / 0 | 미측정 | 미측정 |
| S-QAT-P | D와 동일 | 확보 후 기록 | 미측정 | 예상 82.14% / 0% | 미측정 | 미측정 |

주 손실은 `Acc(D) - Acc(P)`를 pp로, speedup은 `dense median latency / pruned median latency`로 보고한다. 음수 손실도 그대로 기록한다. Dense 정확도 자체가 낮으면 작은 차이만으로 pruning 효과를 과장하지 않는다. Joint 최적화의 우수성은 별도 joint 실험이 있어야 판단할 수 있다.

Pruning 시 backbone parameter 수는 같지만 query 3,840개(D=384, 10 modules)가 추가된다. 학습용 aggregator/head를 추론 parameter로 세지 않는다. 실제 파일 크기와 이론 packed payload는 분리한다.

Batch 1, warmup 10, repeats 100, 같은 validation image와 장치, CUDA synchronize 조건으로 전체 forward latency를 측정한다. Scorer, sorting/gather, LLF cache/복원을 포함한다. GFLOPs는 MAC=2 FLOPs로 dot-product scorer 비용을 포함하고 norm/sort/copy 등 제외 항목을 명시한다. QAT는 fake quantization을 유지하므로 `latency_backend=fake_quant_fp32`, `integer_latency_ms=null`이다. 발표의 RTX 4090을 실제 측정 장치로 가정하지 않는다.

학습 산출물과 query 원본은 보존하고 결과는 별도 평가 폴더에 저장한다. 두 checkpoint hash, split hash, score 방식, router 정밀도, 실제 trace, 라이브러리/장치 정보를 결과에 남긴다.

## 8. 주의사항 / 합의 사항

### 비교 조건과 증강

- A3의 기준은 **앞선 QAT-only 실험의 실제 config/checkpoint**다. 모델, bit 수, pooling, 초기화, 학습 하이퍼파라미터, 증강 조건을 A3에서 새로 정하지 않는다. W4A8·Small과 표의 기본값은 예상 예시다.
- 같은 비교에 묶는 선행 FP32/QAT/joint 실험은 **같은 학습 증강 설정**을 사용해야 한다. 인수 시 Mixup/CutMix/RandomErasing/crop scale을 확인한다. 설정이 다르면 pruning·양자화 효과만의 비교로 해석하지 않고 조건 차이를 따로 기록한다. A3 자체는 재학습하지 않는다.
- **Dense/pruned 정확도 평가는 같은 split·class mapping·평가 전처리로 수행**하며 Mixup/CutMix/RandomErasing을 적용하지 않는다. 같은 checkpoint를 공유하므로 선행 모델의 학습 증강 이력도 두 조건에서 같다.
- Mixup/CutMix를 사용한 학습의 `train_loss`는 soft target 기준이며, Cropr 학습은 auxiliary loss까지 포함할 수 있다. 서로 다른 손실 정의의 `train_loss`로 방법의 우열을 판단하지 않는다. 정확도 비교는 val/test Top-1을 사용하고, A3 추가 손실은 dense Top-1에서 pruned Top-1을 뺀 **pp**로 보고한다.
- 강한 증강이 저비트 QAT 수렴을 어렵게 하거나 pruning 후 정확도 손실을 늘린다는 설명은 **검증이 필요한 가설**이다. 현재 A3 테스트로 학습 수렴이나 정확도 효과를 확인한 것은 아니다. 증강 강도 변경은 선행 실험에서 결정하고 별도 조건으로 기록한다.

### Query와 평가 경계

- Cropr에는 학습된 query가 필요하다. **QAT-only 모델만으로 추가 학습 없는 Cropr를 완성할 수 없다.** Query를 제공받을지, 별도 범위에서 학습할지는 아직 미결정이며 이번 evaluator는 query를 학습하지 않는다.
- Query 학습 데이터·증강·loss·seed·학습 중 pruning 여부와 backbone/observer 고정 여부를 기록한다. Query의 학습 효과가 포함된다는 사실을 명시하고, 학습하지 않은 query나 다른 backbone의 query를 무조건 호환된다고 취급하지 않는다.
- A3 평가에서 backbone·observer·query를 갱신하지 않는다. Fake quantization은 유지하며 재보정이나 optimizer step은 수행하지 않는다. Test split으로 query/R/위치/checkpoint를 고르지 않는다.
- Query bundle의 학습 완료 표시와 provenance는 제공자의 이력 선언이다. Hash/shape/config 검사는 결합 조건을 확인하는 것이며, **query가 실제 학습됐다는 사실을 파일만으로 증명하지는 않는다.**
- R=16에서 첫 단계는 17개다. 12-block/224/patch16 조건의 최대 TPR은 82.14%지만 LLF 복원 후 마지막 block 입력 TPR은 0%다. 이 수치를 전체 연산량 감소율이나 실제 speedup으로 대신 사용하지 않는다.

### 재현성·호환성·측정 해석

| 항목 | 현재 확인된 범위 | 실제 실험에서 남길 사항 |
|---|---|---|
| 선행 설정 상속 | A3 fixture에서 비기본 LR/epochs/label smoothing/bit/experiment 유지 및 평가 batch 분리 확인 | 실제 checkpoint config와 두 조건의 공통 backbone/split hash |
| 증강 이전 checkpoint | 기존 CPU 테스트에서 증강 키 누락 시 off 기본값 보완, 증강 설정 불일치 거부 확인 | 실제 인수 파일의 코드 버전·증강 설정. 다른 구현/pretrained 모델은 loader 지원 여부 확인 |
| Mixup RNG 재개 | 기존 `test_resume_reproduces_mixup_randomness`가 소규모 CPU fixture에서 NumPy RNG 저장·복원을 확인 | 실제 장기 학습·다른 장치/버전까지 동일하다고 보장하지 않음. A3는 학습을 재개하지 않음 |
| A3 재현성 | 같은 sample/seed 사용, 원본 Cropr forward 동등성, 평가 전후 state 불변과 실제 trace 검사 | 라이브러리·GPU·thread·측정 입력 기록. 동점에서 원본 `stable=False`를 사용하므로 장치 간 동일 token 선택을 보장하지 않음 |
| 속도와 용량 | Fake-quant FP32 forward 및 router 포함 지표를 측정하도록 구현 | 실제 저비트 kernel 속도와 구분. Query/cache/LLF 비용 포함, 파일 크기와 packed 추정 분리 |
| 실제 정확도 | 합성 fixture와 ViT-S W4A8 forward 검증만 완료 | Food-101 정확도·수렴·실제 장치 speedup은 미측정. 입력 모델/query 인수 후 검증 |

기존 테스트 근거는 [증강 테스트](../../tests/test_augment.py), A3 근거는 [평가 테스트](../../A3/tests/test_eval_pruning.py)와 [pruning 테스트](../../A3/tests/test_inference_pruning.py)다. 검증한 환경과 건너뛴 GPU 항목은 [실행 문서](execution.md)에 기록한다.

코드는 `A3/`, 문서는 `docs/A3/`에서 관리하고 기존 ViT/QAT/Cropr 코드는 변경하지 않는다. 위 조건이나 query 학습 범위를 바꾸면 실험 설계·실행 config·결과 해석을 함께 갱신한다.
