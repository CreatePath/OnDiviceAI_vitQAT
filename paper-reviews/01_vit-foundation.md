# ViT paper review

**논문:** Alexey Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,” ICLR 2021  
**원문:** [PDF](../papers/05_qat_token_pruning/2021_Dosovitskiy_ViT.pdf)  
**리뷰 상태:** 1차 리뷰

## 한 문장 요약

이미지를 고정 크기 패치의 토큰열로 바꾸고, 거의 표준 Transformer encoder만으로 분류할 수 있음을 보인 ViT의 기반 논문이다. 성능의 핵심 조건은 큰 데이터로 사전학습한 뒤 목표 데이터에 전이하는 것이다.

## 문제와 방법

픽셀 전체에 self-attention을 적용하면 시퀀스가 너무 길어지므로, 입력 이미지를 `P×P` 패치로 나눈다. 각 패치를 선형 투영해 임베딩하고, 학습 가능한 위치 임베딩과 `[CLS]` 토큰을 더해 Transformer encoder에 넣는다. Encoder는 pre-LayerNorm, multi-head self-attention, GELU를 쓰는 MLP, residual 연결로 구성된다. 분류 시 마지막 `[CLS]` 표현을 head에 전달한다.

패치 수는 대략 `N = H×W/P²`다. 패치 크기를 줄이면 공간 정보는 세밀해지지만 토큰 수와 attention 계산량이 급격히 늘어난다. 이 관계가 이후 token pruning 연구의 출발점이다.

## 주요 근거와 결과

저자들은 ImageNet-21k(약 1,400만 장)와 JFT-300M(약 3억 장) 사전학습을 비교했다. 대규모 사전학습에서 ViT-H/14는 ImageNet 88.55%, CIFAR-100 94.55%, VTAB 77.63%를 보고했고, JFT-300M 기준 사전학습 비용은 2.5k TPUv3-core-days였다. 논문은 큰 사전학습 데이터가 CNN의 locality·translation-equivariance 편향이 적은 ViT의 약점을 보완한다고 설명한다.

## 한계와 해석

- 이 논문은 압축이나 양자화 논문이 아니다. 원래의 대형 사전학습 모델 성능을 그대로 온디바이스 소형 ViT에 기대하면 안 된다.
- 중간 규모 데이터만으로 큰 ViT를 학습할 때 과적합 및 inductive-bias 부족이 문제가 된다.
- 제시된 효율성은 대규모 TPU 사전학습 비교 중심이며, 현재 프로젝트의 기기별 추론 지연을 직접 보장하지 않는다.

## 우리 프로젝트와의 연결

ViT-Tiny에서 입력 크기·patch size를 고정하고 token 수를 먼저 기록해야 pruning 결과를 해석할 수 있다. Token Cropr 적용 전후 FLOPs 변화는 남은 토큰 수만으로 추정하지 말고, pruning이 적용되는 block 위치도 함께 기록한다. Food-101에서 사전학습 checkpoint와 fine-tuning 조건을 고정해 양자화 효과와 데이터 조건을 분리해야 한다.

## 다음 확인 질문

1. 현재 실험의 ViT-Tiny patch size와 실제 token 수는 얼마인가?
2. pretrained checkpoint 및 Food-101 fine-tuning 설정은 모든 비교군에서 동일한가?
3. token pruning으로 줄어드는 연산이 실제 추론 환경의 latency에도 반영되는가?
