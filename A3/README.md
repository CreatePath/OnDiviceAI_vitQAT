# A3: QAT-only 모델의 추론 Cropr 실험

실험 코드와 테스트를 기존 `vitqat/`, `cropr/`, `tests/`에서 분리했다. 기존 모델 생성·체크포인트 로더는 수정 없이 import해서 사용한다.

```text
A3/
  eval_pruning.py       # 선행 설정 복원, dense/pruned paired 평가
  inference_pruning.py  # learned-query router, Top-K, LLF
  metrics.py           # A3 전용 FLOPs/parameter/latency 측정
  tests/               # A3 전용 검증
docs/A3/               # 실험 설계, 논문 파라미터, 아키텍처, 실행 문서
```

실험 조건과 실행 방법은 [실험 문서](../docs/A3/README.md)를 따른다. 저장소 루트에서 실행한다.

```powershell
python -m A3.eval_pruning --help
python -m pytest A3/tests -q
```

실제 평가는 선행 QAT-only checkpoint와 학습된 query bundle이 필요하다. Hyperparameter는 선행 config를 상속하며, 기본 pruning은 R=16·첫 단계 17개·LLF on이다.
