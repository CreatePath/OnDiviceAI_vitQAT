# QUOTA paper review

**논문:** Xinqing Li et al., “Towards Joint Quantization and Token Pruning of Vision-Language Models,” arXiv:2604.17320v1, 2026-04-19  
**원문:** [PDF](../papers/05_qat_token_pruning/2026_Li_QUOTA.pdf)  
**리뷰 상태:** 1차 리뷰

## 한 문장 요약

VLM에서 저비트 양자화와 visual-token pruning을 순서대로 따로 적용하면 calibration과 실제 실행 그래프가 달라져 성능이 더 나빠질 수 있다고 보이고, 양자화된 실행 상태에서 pruning 예산과 토큰 점수를 함께 정하는 QUOTA를 제안한다.

## 방법

QUOTA는 calibration set에서 full-precision과 저비트 activation의 상대 오차를 layer별 민감도로 계산한다. 초반·마지막 layer는 후보에서 제외하고, 나머지 layer 중 민감도가 낮은 곳에 더 많은 pruning 예산을 준다. 각 layer의 토큰 점수는 activation magnitude, text-to-visual attention, visual self-attention, quantization error 신호를 robust normalization 후 결합해 deterministic Top-K로 선택한다. 점수 계산과 pruning은 배포와 같은 W4A4 연산 및 양자화 KV cache에서 실행하고, token index·position·mask·cache 정렬을 함께 갱신한다.

## 실험과 결과

주 실험은 LLaVA-1.5-7B, W4A4와 4-bit KV cache, 128개 multimodal calibration sample을 사용한다. Visual token을 최종 30%만 남기는 조건에서 average retention은 95.65%였다. 같은 양자화/ pruning 조건의 stage-wise 조합은 quantize-then-prune 94.30%, prune-then-quant 94.34%였다. Qwen2.5-VL-7B 보조 실험의 30% keep 결과는 93.38% retention이다. 진짜 INT4 CUDA runtime은 모델 크기·peak GPU memory·전력 감소를 확인했지만, pruning에 따른 end-to-end latency는 직접 표로 검증하지 않는다. 보고된 GFLOPs 비율도 저자들이 prefill 중심의 근사 지표라고 명시한다.

## 한계와 해석

- VLM용 post-training calibration 방법이며 ViT-Tiny 이미지 분류 QAT 방법은 아니다.
- 주 실험은 거대 LLaVA 계열, W4A4/KV4, 128개 calibration sample이다. W4A8 Food-101로 결과를 옮겨 말할 수 없다.
- 실제 INT4 경로의 메모리·전력 검증과 pruning의 latency 개선은 구분해야 한다. 저자도 FLOPs 비율을 end-to-end runtime으로 보지 말라고 명시한다.
- 코드가 공개됐다고 적혀 있지 않고, custom CUDA runtime을 사용하므로 재현 비용이 높을 수 있다.

## 우리 프로젝트와의 연결

직접 구현 레시피보다 “양자화와 pruning을 순차 적용하면 calibration/실행 조건이 어긋날 수 있다”는 비교 가설의 근거로 활용한다. 우리 실험은 FP32, QAT-only, Cropr-only, QAT→Cropr, pruning-aware QAT를 나누고 calibration set, pruning schedule, keep ratio를 통제해야 한다. QUOTA의 layer 민감도 기반 예산 분배는 후속 ablation 아이디어다.

## 다음 확인 질문

1. W4A8에서 QAT 뒤 pruning을 적용하면 activation 분포와 scale이 얼마나 변하는가?
2. pruning-aware QAT가 단순 순차 적용보다 Food-101 정확도·실제 latency에서 이득을 주는가?
3. 기기에서 Top-K와 토큰 압축을 지원하지 않는다면 FLOPs 감소가 실제 처리시간으로 이어지는가?
