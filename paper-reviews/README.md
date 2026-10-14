# Paper reviews

ViT 양자화·토큰 pruning 프로젝트의 논문 PDF와 리뷰 노트입니다. 원문은 `papers/05_qat_token_pruning/`에 두고, 리뷰는 이 폴더에 논문별 Markdown으로 기록합니다.

## 확인된 PDF와 1차 리뷰

1. [ViT 원논문 리뷰](01_vit-foundation.md) — [PDF](../papers/05_qat_token_pruning/2021_Dosovitskiy_ViT.pdf)
2. [PTQ-ViT 리뷰](02_ptq-vit.md) — [PDF](../papers/05_qat_token_pruning/2021_Liu_PTQ_ViT.pdf)
3. [Token Cropr 리뷰](03_token-cropr.md) — [PDF](../papers/05_qat_token_pruning/2025_Bergner_Token_Cropr.pdf)
4. [QUOTA 리뷰](04_quota.md) — [PDF](../papers/05_qat_token_pruning/2026_Li_QUOTA.pdf)

## 권장 리뷰 흐름

ViT 구조와 토큰 수 → PTQ의 ViT 특수 이슈 → Cropr로 token pruning 이해 → QUOTA로 저비트 실행과 pruning의 상호작용 비교.

현재 원문 폴더에는 위 4편이 있습니다. 실험 계획 README에 언급된 LSQ, Q-ViT, DynamicViT 원문은 아직 이 폴더에서 확인되지 않았습니다.

## 현재까지의 종합 메모

프로젝트 목표(README 기준)는 ViT-Tiny, Food-101, W4A8 QAT와 token pruning을 함께 평가하는 것입니다. 네 논문 중 직접적으로 quantization-aware training을 다루는 것은 없습니다: PTQ-ViT와 QUOTA는 post-training 계열이고, Cropr는 pruning 학습을 다루되 저비트 양자화와 결합하지 않습니다. 따라서 QAT 방법 논문(예: LSQ와 Q-ViT)을 보충해야 QAT baseline을 정당화할 수 있습니다.

비교군은 우선 FP32 baseline, QAT-only, Cropr-only, QAT 후 Cropr 순차 적용으로 분리하고, 가능하면 pruning-aware QAT를 추가합니다. Dataset split·checkpoint·bit-width·keep ratio·pruning 위치를 통제하며 정확도, FLOPs, 모델 크기, 실제 기기 latency를 모두 기록해야 합니다.
