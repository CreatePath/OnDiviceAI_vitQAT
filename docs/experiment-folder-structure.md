# 실험 코드와 문서의 독립 폴더 구조

새 실험은 저장소 루트의 **`<실험 ID>/`에 코드**, **`docs/<실험 ID>/`에 문서**를 둔다. 현재 적용한 실험 ID는 `A3`다. 기존 ViT·QAT 기준선과 Cropr 원본은 그대로 유지하고, 실험별 동작은 각 실험 폴더에서 구현한다.

## 현재 A3 구조

```text
저장소 루트/
  vitqat/                         # 기존 ViT·QAT 공통 코드
  cropr/                          # 기존 Cropr 원본
  tests/                          # 기존 공통 코드의 회귀 테스트
  A3/                             # A3 실험 전용 코드
    __init__.py
    README.md                     # 코드 진입 안내와 실험 문서 링크
    eval_pruning.py               # 설정 상속, checkpoint 로드, paired 평가
    inference_pruning.py          # 이전 import 경로 호환
    pruning/                      # import해서 재사용하는 학습/추론 API
    metrics.py                    # A3 전용 측정
    tests/                        # A3 전용 테스트와 fixture
    checkpoints/                  # 인수한 모델/query, 필요 시 로컬 생성
    runs/                         # A3 평가 결과, 필요 시 로컬 생성
  docs/
    experiment-folder-structure.md # 이 폴더 구조 규칙
    experiments.md                # 공통 실험 실행 안내와 개별 실험 링크
    A3/                           # A3 실험 전용 문서
      README.md                   # 목적, 범위, 문서 목록
      experiment-design.md        # 비교군, 상속 조건, 평가 절차
      pruning-parameters.md       # 논문 근거와 pruning 설정
      architecture.md             # 모듈 책임과 인터페이스
      execution.md                # 실행 명령과 입력/출력 계약
```

`A3/checkpoints/`와 `A3/runs/`는 코드 실행이나 파일 인수 시 만드는 로컬 경로 예시다. 현재 학습된 모델이나 실험 결과가 이미 들어 있다는 뜻은 아니다. 대용량 checkpoint·데이터·결과는 `A3/.gitignore`에서 제외한다.

## 배치 기준

| 구분 | 위치 | 관리 원칙 |
|---|---|---|
| 실험 진입점, 알고리즘, wrapper, 전용 지표 | `<실험 ID>/` | 해당 실험 안에서 구현 |
| 전용 테스트와 fixture | `<실험 ID>/tests/` | 기존 `tests/`와 분리 |
| 실험 설계, 파라미터, 논문 근거 | `docs/<실험 ID>/` | 코드 구조와 구분해 관리 |
| 구현 아키텍처, API, checkpoint 계약 | `docs/<실험 ID>/architecture.md` | 구현 변경과 함께 갱신 |
| 실행 방법과 결과 형식 | `docs/<실험 ID>/execution.md` | 실제 동작하는 명령 기준 |
| 입력 모델과 결과 | `<실험 ID>/checkpoints/`, `<실험 ID>/runs/` | 원본 입력 보존, 출력 별도 저장 |
| 공통 규칙과 실험 목록 | `docs/` 바로 아래 | 실험별 문서로 연결 |

추가 실험도 같은 규칙으로 폴더를 만든다. 예를 들어 A4를 시작하면 `A4/`와 `docs/A4/`를 함께 생성한다. 현재 A1/A2/A4 폴더가 존재한다고 가정하지 않는다.

## 기존 코드와의 경계

각 실험은 기존 모델 생성기, checkpoint 로더, 데이터 loader를 **수정 없이 import해서 재사용**할 수 있다. 폴더 독립은 공통 코드를 모두 복사한다는 뜻이 아니다. 예를 들어 A3는 `vitqat.models.build_model`로 선행 모델을 생성하고 A3 wrapper로 추론 pruning을 추가한다.

새 실험의 scorer, forward 변경, 지표 계산을 위해 `vitqat/`나 `cropr/` 원본을 덮어쓰지 않는다. 실험 전용 동작은 wrapper·adapter·별도 진입점으로 구현하고, 테스트도 해당 실험 폴더에 둔다. 공통 코드 자체의 수정이 필요하다면 실험 전용 구현과 구분된 공통 변경으로 관리한다.

## 설정과 재현 조건

A3는 **앞선 QAT-only 실험의 최종 config/checkpoint**를 기준으로 모델, 데이터 분할, 전처리, pooling, 학습 하이퍼파라미터와 QAT 설정을 상속한다. A3에서는 추론 Cropr 설정만 추가한다. 문서의 기본값보다 실제 선행 설정이 우선한다.

선행 설정을 그대로 지원하지 못하면 지원 범위를 명시하고 adapter를 작성한다. 다른 기본 모델이나 정밀도로 조용히 대체하지 않는다. 평가 장치·batch 등 환경 변경은 명시적으로 기록하고 dense/pruned 양쪽에 동일하게 적용한다.

결과에는 선행 config, 실제 평가 config, checkpoint/query SHA256, split hash, pruning 설정, 측정 환경을 저장한다. Query가 없으면 임의로 생성하지 않으며, 학습된 query 확보는 [A3 입력 계약](A3/execution.md)을 따른다.

## 실행 위치와 검증

모든 명령은 저장소 루트에서 실행한다. Python package 경로로 실행해 경로에 따른 import 차이를 줄인다.

```powershell
python -m A3.eval_pruning --help
python -m pytest A3/tests -q
python -m pytest tests A3/tests -q
```

첫 번째 테스트 명령은 A3만, 두 번째는 기존 코드 회귀와 A3를 함께 확인한다. A3 전체 실행은 [실행 문서](A3/execution.md), 설계는 [실험 문서 목록](A3/README.md)을 참고한다.
