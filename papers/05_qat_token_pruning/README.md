# QAT와 Token Pruning 공동 학습 읽기 목록

대상 발표: `온디바이스AI 중간발표.pptx`의 ViT-Tiny, Cropr, Food-101 실험. 목표는 W4A8 QAT 환경에서 token pruning을 학습 단계부터 같이 최적화할 때 생기는 효과와 한계를 이해하는 것이다.

## 추천 읽기 순서

ViT → PTQ-ViT → LSQ → Q-ViT → DynamicViT → Token Cropr → QUOTA

## 먼저 구분할 개념

| 개념 | 핵심 질문 | 실험에서의 의미 |
|---|---|---|
| Quantization 방식 | 무엇을 몇 bit로, 어떤 scale 단위로 표현할까? | W4A8은 weight 4-bit, activation 8-bit를 뜻한다. symmetric/asymmetric, per-tensor/per-channel도 함께 비교한다. |
| PTQ | 학습을 끝낸 FP32 모델을 어떻게 저정밀도로 바꿀까? | calibration data로 weight/activation의 scale·zero-point 또는 interval을 정하고, gradient update는 하지 않는다. |
| QAT | 양자화 오차를 보면서 어떻게 학습할까? | forward에서 fake quantization을 넣고 STE로 gradient를 근사하여 weight와 scale을 fine-tune한다. |

| 순서 | 논문 | 연도 | 발표와의 연결 | 핵심 확인 항목 | PDF |
|---:|---|---:|---|---|---|
| 1 | An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale — Dosovitskiy et al. | 2021 | ViT-Tiny의 patch token, [CLS], MHSA의 출발점 | token 수가 attention FLOPs에 미치는 영향 | [2021_Dosovitskiy_ViT.pdf](2021_Dosovitskiy_ViT.pdf) |
| 2 | Post-Training Quantization for Vision Transformer — Liu et al. | 2021 | PTQ가 calibration으로 quantization interval을 찾는 실제 절차를 익힌다. | weight/input interval, ranking loss, mixed precision | [2021_Liu_PTQ_ViT.pdf](2021_Liu_PTQ_ViT.pdf) |
| 3 | Learned Step Size Quantization — Esser et al. | 2020 | QAT의 fake quantization과 학습 가능한 scale을 이해하는 기반 | STE, quantization step size의 gradient | [2020_Esser_LSQ.pdf](2020_Esser_LSQ.pdf) |
| 4 | Q-ViT: Fully Differentiable Quantization for Vision Transformer — Li et al. | 2022 | ViT에서 QAT가 attention/GELU와 만나 생기는 특수한 문제 | head-wise robustness, scale·bit-width 공동 학습 | [2022_Li_Q_ViT.pdf](2022_Li_Q_ViT.pdf) |
| 5 | DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification — Rao et al. | 2021 | pruning score와 top-k mask를 학습에 넣는 선행 방법 | hierarchical pruning, differentiable attention masking | [2021_Rao_DynamicViT.pdf](2021_Rao_DynamicViT.pdf) |
| 6 | Token Cropr: Faster ViTs for Quite a Few Tasks — Bergner et al. | 2025 | 발표에서 실제 token pruner로 쓰는 Cropr의 원 논문 | learnable query, auxiliary head, task-aware token selection | [2025_Bergner_Token_Cropr.pdf](2025_Bergner_Token_Cropr.pdf) |
| 7 | Towards Joint Quantization and Token Pruning of Vision-Language Models — Li et al. (QUOTA) | 2026 | 발표의 quantization-pruning mismatch 및 공동 최적화 가설의 직접 근거 | deployed low-bit 환경의 token importance, budget allocation, stage-wise 대비 | [2026_Li_QUOTA.pdf](2026_Li_QUOTA.pdf) |

## 발표 실험에 적용할 때의 체크포인트

- QUOTA는 VLM의 visual-token pruning 논문이다. ViT-Tiny/Food-101에서 같은 결론을 가정하지 말고, QAT-only, QAT 후 top-k, pruning 후 PTQ, joint QAT+Cropr를 같은 token budget과 W4A8 설정에서 비교한다.
- Cropr의 auxiliary loss와 query 수, pruning layer, keep ratio를 고정한 뒤 QAT 유무만 바꿔야 공동 학습의 효과를 분리할 수 있다.
- 정확도 외에 실제 latency, GFLOPs, token 유지율, 모델 저장 용량을 함께 기록한다. 낮은 FLOPs가 항상 GPU latency 감소로 이어지지는 않는다.

## 공식 PDF 출처

- [ViT](https://arxiv.org/pdf/2010.11929), [PTQ-ViT — NeurIPS 2021](https://papers.nips.cc/paper_files/paper/2021/file/ec8956637a99787bd197eacd77acce5e-Paper.pdf), [LSQ](https://arxiv.org/pdf/1902.08153), [Q-ViT](https://arxiv.org/pdf/2201.07703)
- [DynamicViT — NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/747d3443e319a22747fbb873e8b2f9f2-Paper.pdf), [Token Cropr — CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Bergner_Token_Cropr_Faster_ViTs_for_Quite_a_Few_Tasks_CVPR_2025_paper.pdf), [QUOTA](https://arxiv.org/pdf/2604.17320)
