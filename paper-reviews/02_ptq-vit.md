# PTQ-ViT paper review

**논문:** Zhenhua Liu et al., “Post-Training Quantization for Vision Transformer,” NeurIPS 2021  
**원문:** [PDF](../papers/05_qat_token_pruning/2021_Liu_PTQ_ViT.pdf)  
**리뷰 상태:** 1차 리뷰

## 한 문장 요약

ViT를 추가 학습 없이 양자화할 때 attention의 순위 정보를 보존하는 손실과 layer별 혼합 비트 배정을 사용해 일반적인 균일 PTQ보다 정확도 저하를 줄인다.

## 방법

PTQ calibration set에서 weight와 activation의 quantization interval을 교대로 탐색한다. 일반 선형층은 원본 출력과 양자화 출력의 유사도를 맞추고, self-attention에는 ranking loss를 더해 attention 값의 상대적 순위가 바뀌는 현상을 억제한다. Bias correction은 여러 층을 지날 때 누적되는 오차를 줄인다. 마지막으로 attention map과 MLP 출력의 nuclear norm을 layer 민감도 지표로 사용해 혼합 정밀도 비트를 배정한다.

중요한 구현 조건: 논문은 행렬곱의 가중치와 입력을 양자화하지만 Softmax와 LayerNorm은 양자화하지 않는다. 따라서 결과는 완전한 정수 전용 ViT 실행 결과가 아니다.

## 실험과 결과

CIFAR-10/100, ImageNet, COCO에서 ViT/DeiT/DETR을 평가했다. calibration은 CIFAR에서 100장, ImageNet 및 COCO에서 1,000장을 사용했다. DeiT-B ImageNet의 8-bit 혼합 정밀도 결과는 81.29%로, FP 기준 81.8%에 근접했다. ViT-B ImageNet에서는 8-bit 혼합 정밀도 76.98%, FP 77.91%를 보고했다. 저자들은 Hessian 기반 혼합 정밀도보다 훨씬 적은 탐색 시간을 보고한다.

## 한계와 해석

- PTQ이며 QAT가 아니다. 목표 W4A8 QAT와 학습 절차가 다르다.
- Softmax와 LayerNorm을 남겨둬 완전 정수 연산 배포 문제를 해결하지 않는다.
- 논문은 accuracy, 압축 크기와 추정 연산량을 주로 비교하며, 목표 기기의 end-to-end latency를 입증하지 않는다.
- 모델·데이터셋이 현재 ViT-Tiny/Food-101 실험과 다르므로 논문 수치를 직접 성능 목표로 삼으면 안 된다.

## 우리 프로젝트와의 연결

QAT 결과 해석을 위한 비교 기준으로 활용한다. 최소한 FP32, naive PTQ, QAT-only를 같은 모델·데이터 분할·평가 코드로 비교하고, weight/activation bit-width와 LayerNorm/Softmax의 실제 정밀도를 명시해야 한다. PTQ-ViT의 ranking-aware 아이디어가 token importance 순위 보존에 주는 시사점은 가설로 둘 수 있지만, 이 논문에서 pruning을 실험한 것은 아니다.

## 다음 확인 질문

1. 현재 목표 W4A8의 양자화 단위(per-tensor/per-channel)와 scale 학습 방식은 무엇인가?
2. Fake quant를 쓴 성능과 실제 integer kernel 성능을 별도로 확인할 수 있는가?
3. QAT 과정에서 attention 순위 또는 pruning score가 얼마나 바뀌는가?
