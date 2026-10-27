# W2 가중치를 유지하는 TensorRT INT8 실행

기존 W2A8 QAT 체크포인트를 TensorRT용 explicit INT8 Q/DQ ONNX로 내보낸다.
Linear/Conv뿐 아니라 backbone attention의 Q/K/V, softmax 확률에도 INT8 Q/DQ를 넣는다.
TensorRT가 실제로 선택한 커널은 엔진의 레이어 보고서로 별도 확인해야 한다.
ONNX export 성공만으로 전체 attention INT8 실행이나 속도 향상이 검증된 것은 아니다.

## 무엇이 유지되고 바뀌는가

| 항목 | 동작 |
| --- | --- |
| W2 가중치 | 기존 scale과 zero-point로 먼저 양자화한 값을 고정. 값은 유지하고 INT8에 담는다. |
| 가중치 표현 | `q8 = q2 - zero_point`, `w = scale * q8`. 2비트 packing이나 INT2 커널은 사용하지 않는다. |
| A8 | 기존 affine 범위에서 대칭 signed INT8 범위를 계산한다. 기존 A8 반올림 결과와 같지는 않다. |
| Attention | Q/K/V와 확률에 A8 Q/DQ를 적용. QK 및 확률×V의 두 입력이 Q/DQ에 연결됐는지 검사한다. |
| Softmax/LayerNorm/GELU/residual | 부동소수점 연산. INT8 attention은 softmax 자체까지 정수라는 뜻이 아니다. |
| Cropr | `qat_scope=all`은 scorer QK도 A8 대상으로 내보낸다. `backbone`은 기존대로 Cropr를 FP32로 유지한다. |
| 원본 모델/체크포인트 | 복사본만 변환. 기존 학습과 BitBLAS 경로는 유지한다. |

TensorRT INT8의 zero-point는 0이어야 한다. activation의 기존 scale이 `s`,
unsigned zero-point가 `z`일 때 다음 scale을 사용한다.

```text
s_new = s * max(z, 255 - z) / 127
q_new = clamp(round(x / s_new), -128, 127)
```

이는 기존 표현 범위를 덮는 초기 변환이며 **새 데이터로 calibration하거나 재학습하지 않는다.**
특히 음수가 없는 softmax 확률은 signed INT8의 양수 구간만 사용하므로 기존 UINT8보다
유효 양수 단계가 줄어든다. 정확도가 떨어지면 대칭 A8에 맞춘 calibration/QAT가 추가로 필요하다.
기존 W2A8 정확도가 그대로 유지된다고 보고하면 안 된다.

QK의 attention scale은 Q의 Q/DQ 이전으로 옮기고 Q의 양자화 scale에도 같은 값을 곱한다.
이렇게 하면 행렬곱 직전의 별도 부동소수점 곱셈을 피할 수 있다.
Cropr 정렬은 export 사본에서 `stable=False`를 생략해 ONNX exporter가 지원하는 overload를 쓴다.
기본값은 동일하며 upstream Cropr 소스는 변경하지 않는다. 동점 순서는 런타임 간 달라질 수 있다.

## 준비

기존 CUDA PyTorch 학습 환경에서 아래 명령으로 필요한 Python 패키지를 설치한다.
TensorRT Python bindings와 CUDA 12용 runtime도 함께 설치된다.
`trtexec`, 별도 SDK 다운로드, 실행 파일 경로 설정은 필요 없다.

```bash
python -m pip install -r requirements-tensorrt.txt
python -m pytest tests/test_tensorrt_int8.py -q
```

TensorRT는 `tensorrt-cu12==10.13.3.9`로 고정한다. NVIDIA 드라이버와 CUDA-enabled PyTorch는 필요하다.
TensorRT-RTX나 TensorRT 11은 이 실행 경로의 대상이 아니다.
CPU에서 export만 할 경우에는 `python -m pip install onnx==1.17.0 onnxruntime==1.20.1`이면 된다.

체크포인트는 `--mode qat --weight-bits 2 --activation-bits 8`로 학습하고 observer 상태가 채워져 있어야 한다.
모델 설정은 체크포인트에서 복원한다. W2A2, 비활성 fake quant, 미초기화 observer는 거부한다.

## CPU에서 ONNX 내보내기

```bash
python -m scripts.benchmark_tensorrt_int8 \
  --checkpoint runs/w2a8/best.pt \
  --output runs/tensorrt_int8 --batch-size 1 --export-only
```

`fp32_master.onnx`, `w2_int8.onnx`, 동일한 입력을 담은 `images.bin`, `report.json`이 생성된다.
batch와 이미지 크기는 고정이다. 다른 batch로 실행하려면 해당 batch로 다시 export/build한다.
`fp32_master`는 같은 체크포인트의 master parameter를 사용한 latency 기준선이며,
별도로 학습된 FP32 정확도 기준선은 아니다.

## GPU에서 엔진 생성과 속도 비교

```bash
python -m scripts.benchmark_tensorrt_int8 \
  --checkpoint runs/w2a8/best.pt \
  --output runs/tensorrt_int8 --batch-size 1 \
  --warmup-ms 1000 --duration 10 --iterations 100
```

두 ONNX를 TensorRT Python API로 각각 엔진으로 만들고 동일 입력으로 측정한다.
GPU 선택은 `--device cuda:0` 등으로 지정한다. 빌드 진단은 터미널에 출력하며,
실패 시 report에 오류를 기록하고 종료한다. 다른 backend로 fallback하지 않는다.
명시적 Q/DQ를 사용하며 별도의 무작위 calibration은 수행하지 않는다.
FP16도 켜지 않는다. 실제 INT8로 선택되지 않은 연산은 FP32로 남을 수 있다.

- `*.engine`: 해당 GPU/TensorRT 환경용 엔진.
- `*.layers.json`: 실제 엔진 레이어, tensor format/type, tactic 정보.
- `*.times.json`: warmup 이후 GPU 측정 원자료.
- `*.outputs.json`: 합성 입력에 대한 실제 엔진 logits.
- `report.json`: 변환 전후 합성 logits 차이, ONNX attention Q/DQ 검사,
  TensorRT/CUDA 버전, 장치, 측정 설정, native/reference logits 차이, GPU median/p95와 속도비.

`gpu_compute_speedup_vs_trt_fp32 > 1`이면 해당 조건에서 TensorRT FP32보다 빠르다.
이 값은 PyTorch FP32 대비 속도비가 아니다. 두 엔진 모두 TF32를 끄고 동일 방식으로 측정한다.
PyTorch CUDA tensor의 주소를 TensorRT에 연결하고 같은 CUDA stream에서 실행한다.
입력 전송과 첫 실행·warmup을 마친 뒤 각 forward를 CUDA event로 재고 매번 동기화한다.
데이터 로딩/전송과 빌드 시간은 측정에서 제외하고 CUDA Graph는 켜지 않는다.
이 값은 host latency가 아니며 Python enqueue 지연이 CUDA event 구간에 일부 포함될 수 있다.
이전 `trtexec` 측정과도 enqueue 방식이 달라 직접 같은 지표로 취급하면 안 된다.

`w2_int8.layers.json`에서 backbone attention의 두 BMM 또는 해당 fused attention 레이어의
입력 format/type 및 tactic을 확인한다. INT8 Q/DQ나 reformat 레이어가 존재한다는 것만으로는
INT8 행렬곱을 입증할 수 없다. FP32 fallback 또는 정밀도가 불명확한 fused layer가 있으면
"전체 INT8 attention 검증 완료"로 표시하지 않는다. `report.json`도 이 판단을 자동 통과시키지 않는다.

## 정확도 평가와 검증 범위

`convert_tensorrt_int8(qat_model)`은 CPU PyTorch 수치 참조 모델을 반환한다.
기존 `vitqat.training.evaluate`와 validation loader에 연결해 원본 QAT와 비교할 수 있다.
이 참조 모델 자체는 정수 커널로 가속되지 않는다. 최종 배포 정확도는 실제 엔진으로 평가해야 한다.
합성 입력의 top-1 일치율은 Food-101 정확도를 대체하지 않는다.

테스트는 W2 값 보존과 원본 불변성, 잘못된 상태 거부, Cropr off/on 및 LLF/scope 조합,
ONNX checker, attention 입력의 Q/DQ 연결, ONNX Runtime 출력, CLI export를 검증한다.
12개 블록을 거치며 런타임 간 작은 행렬곱 오차가 양자화 경계를 넘을 수 있어
전체 출력 검증은 bitwise 비교가 아니라 최대 8 output bins 및 상대 L2 5% 미만의 drift 검사다.
이 한계는 학습된 모델의 허용 정확도 기준을 뜻하지 않는다.

로컬에는 학습 체크포인트와 TensorRT 실행 환경이 없어 실제 엔진 빌드, INT8 커널 선택,
RTX 4090 속도 및 Food-101 정확도는 아직 검증하지 않았다. 가속 여부는 위 GPU 명령으로 확인해야 한다.

2026-10-28 최초 export 구현 검증 결과:

- PyTorch 2.14.1 CPU 환경: 전체 `tests` 75 passed, 30 skipped. CUDA 항목은 실행하지 못했다.
- 별도 PyTorch 2.3.0 CPU / torchvision 0.18.0 CPU 환경: 신규 테스트 8 passed.
- ONNX 1.17.0 / ONNX Runtime 1.20.1로 실제 export 및 CPU 실행 검증.
- 테스트 체크포인트는 무작위 초기화 모델의 observer를 합성 입력으로 채운 것이며 학습된 모델이 아니다.

Python API 실행 경로에는 FP32/INT8 엔진을 실제로 생성·실행하는 native 테스트를 추가했다.
CUDA PyTorch 또는 TensorRT가 없는 환경에서는 해당 테스트를 건너뛰며, 통과로 간주하지 않는다.
Python API 전환 후 PyTorch 2.3 CPU에서 관련 테스트는 9 passed, native 2 skipped였다.

참고: [TensorRT explicit quantization](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html),
[엔진 검사 도구](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-tools.html),
[TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html).
