# Token Cropr paper review

**논문:** Benjamin Bergner, Christoph Lippert, Aravindh Mahendran, “Token Cropr: Faster ViTs for Quite a Few Tasks,” CVPR 2025  
**원문:** [PDF](../papers/05_qat_token_pruning/2025_Bergner_Token_Cropr.pdf)  
**리뷰 상태:** 1차 리뷰

## 한 문장 요약

보조 예측 head의 학습 신호로 task-relevant patch를 고르고, 이후 ViT block에서 덜 중요한 토큰을 줄이는 token pruning 방법이다. 분류뿐 아니라 dense prediction에서는 마지막 block에서 잘린 토큰을 다시 합류시키는 LLF를 제안한다.

## 방법

Cropr 모듈은 learnable query 기반 cross-attention으로 patch 점수를 계산하고 Top-K로 유지 토큰을 선택한다. 학습 때는 aggregator와 task-specific auxiliary head가 토큰 점수가 본래 task에 유용하도록 신호를 준다. Stop-gradient로 auxiliary loss가 backbone 표현을 직접 흔들지 않게 한다. 추론 때 auxiliary head와 aggregator를 제거하고 가벼운 router만 남긴다.

분류에서는 하나의 query와 분류 보조 head를 쓴다. Segmentation은 공간 위치별 query를 두고, detection/instance segmentation은 multi-label 보조 과제를 이용한다. Dense task용 Last Layer Fusion(LLF)은 제거한 토큰을 마지막 ViT block 직전에 retained tokens와 다시 합쳐, 모든 이미지 위치의 예측 정보를 복원한다.

## 실험과 결과

ImageNet 분류, ADE20k segmentation, COCO detection/instance segmentation을 평가했고, inference throughput은 NVIDIA A100에서 측정했다. EVA-02-L ImageNet 설정에서 LLF를 적용한 Cropr는 89.9%에서 89.7%로 0.2%p 낮아지는 대신 FLOPs 41% 감소와 약 2.1배 throughput을 보고했다. 더 공격적으로 80% 토큰을 한 번에 제거한 설정은 88.8% 정확도, 약 4.1배 throughput이었다. ADE20k에서는 median mIoU 56.7에서 56.6으로 0.1p 차이, 약 2배 speedup을 보고했다. COCO detection/instance segmentation에서는 63.0 box AP / 54.0 mask AP, encoder 2.4배 및 전체 모델 1.9배 throughput을 보고했다.

## 한계와 해석

- 이 방법은 QAT/quantization을 다루지 않는다. pruning-aware QAT에 그대로 적용되는 성능 보장은 없다.
- Task별 보조 head와 재학습이 필요하다. 발표 실험에서는 분류 설정부터 맞추는 것이 현실적이다.
- 정해진 pruning schedule의 static Top-K라 입력별 난이도에 따라 keep ratio를 바꾸지 않는다.
- 저자 결과는 특정 EVA/ViT 설정과 A100 측정에 기반한다. FLOPs 감소가 목표 기기에서 같은 latency 감소로 이어지는지는 별도 측정해야 한다.

## 우리 프로젝트와의 연결

Food-101 분류에 Cropr를 붙일 때는 먼저 동일 checkpoint의 pruning-only 기준을 만들고, 이어 QAT-only 및 QAT+Cropr 조합을 비교한다. Cropr의 학습 schedule, 유지 토큰 수, pruning block 위치를 고정한 뒤 정확도·FLOPs·실제 latency를 함께 기록해야 양자화와 pruning의 효과를 분리할 수 있다.

## 다음 확인 질문

1. 발표 실험에서 Cropr를 넣는 block과 최종 keep ratio는 무엇인가?
2. Quantization fake-op이 들어간 상태에서 Cropr scorer를 학습할지, scorer를 먼저 학습 후 QAT할지?
3. Cropr의 Top-K가 실제 배포 backend에서 효율적으로 실행되는가?
