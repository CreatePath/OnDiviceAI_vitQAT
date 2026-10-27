# A3: CLS attention 기반 토큰 pruning

실험 코드와 테스트를 기존 `vitqat/`, `cropr/`, `tests/`에서 분리했다. 기존 모델 생성·체크포인트 로더는 수정 없이 import해서 사용한다.

```text
A3/
  eval_pruning.py       # 선행 설정 복원, dense/pruned paired 평가
  inference_pruning.py  # 이전 import 경로 호환
  pruning/             # 공개 라이브러리 API (attach_pruning), 학습/추론 wrapper
  metrics.py           # A3 전용 FLOPs/parameter/latency 측정
  tests/               # A3 전용 검증
docs/A3/               # 실험 설계, 논문 파라미터, 아키텍처, 실행 문서
```

실험 조건과 실행 방법은 [실험 문서](../docs/A3/README.md)를 따른다. 저장소 루트에서 실행한다.

```powershell
python -m A3.eval_pruning --help
python -m pytest A3/tests -q
```

A3 추론 평가는 QAT-only 체크포인트를 사용한다. CLS attention Top-K에는 학습된 query가 필요 없다. 기본 스케줄은 첫 단계 17개, 이후 각 단계 16개를 제거하고 LLF를 적용한다. 학습·추론 라이브러리 사용과 검증 결과는 [구현 기록](../docs/A3/cls-attention-pruning.md)을 참조한다.

다른 학습·추론 코드에서 재사용하려면 `from A3.pruning import attach_pruning`을 사용한다. 호출 방법과 보조 loss 계약은 [라이브러리 API](../docs/library-api.md)에 정리했다. `inference_pruning.py`는 이전 import의 호환 파일이다.
