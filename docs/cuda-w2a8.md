# RTX 4090 GPU W2A8 추론

## BitBLAS 추론

GPU 정수 추론은 **BitBLAS**만 사용한다. `--cuda-w2a8-inference`로 정수 추론을 켠다.
행렬곱 커널은 라이브러리에 맡기고 프로젝트에서는 QAT 변환과 affine 보정을 처리한다.
지원하지 않는 환경에서는 오류를 내며 다른 백엔드로 자동 대체하지 않는다.

Python API도 BitBLAS를 사용한다.

```python
from vitqat.cuda_w2a8 import convert_cuda_w2a8
inference_model = convert_cuda_w2a8(qat_model, "cuda")
```

## Linux RTX 4090 환경

별도 Python 환경에 CUDA Toolkit과 C/C++ 컴파일러를 준비한다.
BitBLAS wheel은 Linux x86_64용이며 공식 설치 기준은 Ubuntu 20.04 이상이다.
프로젝트의 PyTorch 2.3.0+cu121을 유지하고 BitBLAS 0.1.0.post1을 고정한다.
API는 해당 배포 wheel의 소스로 확인했지만, 이 조합의 실제 서버 실행은 아직 검증하지 않았다.

```bash
python -m pip install -r requirements-bitblas.txt
python -c "import torch; from vitqat.w2a8_bitblas import require_bitblas; bitblas = require_bitblas(); assert torch.cuda.is_available(); print(torch.__version__, bitblas.__version__, torch.cuda.get_device_name())"
python -m pytest tests/test_bitblas_w2a8.py -k native -q -rs
```

native 테스트 12개가 모두 실행되는지 확인한다. 건너뛴 테스트는 통과로 간주하지 않는다.
초기 변환과 처음 보는 행렬 크기에서는 라이브러리 컴파일 시간이 발생한다.

### `libnvrtc.so.12`를 찾지 못할 때

BitBLAS에 포함된 TVM은 CUDA 12 NVRTC 공유 라이브러리를 로딩한다.
이 오류는 라이브러리가 설치되지 않았거나 동적 로더의 검색 경로에 없는 경우 발생한다.
PyTorch의 CUDA 실행이 가능하더라도 TVM의 라이브러리 로딩이 성공한다는 뜻은 아니다.

프로젝트의 `require_bitblas()`는 pip 패키지의 NVRTC를 절대 경로로 먼저 로딩한 뒤 BitBLAS를 import한다.
이 경로에는 NVRTC용 `LD_LIBRARY_PATH` 설정이 필요하지 않다. 저장소 루트에서 다음 명령으로 확인한다.

```bash
python -m pip install nvidia-cuda-nvrtc-cu12==12.1.105
python -c "from vitqat.w2a8_bitblas import require_bitblas; print(require_bitblas().__version__)"
```

`python -c 'import bitblas'`는 프로젝트의 사전 로딩을 거치지 않는다. 직접 import하려면
다음 환경 변수를 설정한 뒤 새 Python 프로세스를 시작한다.

```bash
CUDA_NVRTC_LIB_DIR="$(python -c 'from importlib.metadata import distribution; print(distribution("nvidia-cuda-nvrtc-cu12").locate_file("nvidia/cuda_nvrtc/lib"))')"
export LD_LIBRARY_PATH="${CUDA_NVRTC_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python -c "import torch, bitblas; print(bitblas.__version__)"
```

사전 로딩에 실패하면 시도한 절대 경로와 원래 로더 오류를 출력한다.
NVRTC 패키지는 전체 CUDA Toolkit이나 `nvcc`를 대신하지 않는다.

## 양자화 방식과 범위

QAT 체크포인트를 복사해 변환하며 학습된 scale과 zero-point를 그대로 사용한다.
별도 calibration set이나 PTQ 재양자화는 사용하지 않는다. W2/A8이 아니거나
observer 관측값이 없거나 fake quantization이 꺼진 모델은 변환을 거부한다.

- **Linear/Conv:** W2 코드 0~3을 BitBLAS UINT2 형식으로 한 번 패킹한다.
  입력은 기존 A8 격자로 양자화하고 `q_a - 128`을 실제 INT8에 저장한다.
  BitBLAS가 INT8×UINT2 행렬곱을 INT32로 누산한다.
- **비대칭 보정:** 누산값에 `-sum(q_a-128)*z_w + (128-z_a)*sum(q_w-z_w)`를 더한다.
  이어 원래 scale, bias, 출력 A8 반올림·클리핑을 적용하고 FP32로 역양자화한다.
  원래 W2 값을 signed INT2로 다시 양자화하지 않는다.
- **Attention/Cropr:** Q/K/V/확률을 실제 INT8에 저장하고 QK·PV를 BitBLAS INT8×INT8로 계산한다.
  양쪽 zero-point를 보정한다. 고정 Cropr query의 W2 양자화와 Q8 변환은 미리 계산한다.
- **나머지 연산:** LayerNorm, softmax, GELU, residual, 위치 임베딩, pruning은 기존 경로를 유지한다.
  `--qat-scope backbone`이면 QAT 대상이 아닌 Cropr도 FP32를 유지한다.

변환 후 Quantizer/FakeQuantize가 남아 있으면 실패한다. 이 경로는 W2A8 전용이며 W2A2는 지원하지 않는다.

BitBLAS Matmul은 2D이므로 attention의 batch/head별로 호출한다. 런타임 K/V에는 가중치 사전 변환을 적용하지 않는다.
M/N은 16, K는 32의 배수로 패딩하고 출력을 원래 크기로 자른다. 보정에는 원래 차원의 값만 사용한다.
Conv는 unfold 뒤 행렬곱을 수행한다. 현재 fast decoding, layout propagation, autotuning은 끄고
연산자를 크기·dtype·GPU별로 캐시한다. 작은 행렬, 패딩, Python 호출, 양자화·보정 비용 때문에
라이브러리 도입만으로 FP32보다 빨라진다고 보장할 수 없다.

## 체크포인트 평가

```bash
python -m vitqat.train \
  --device cuda --data-root data --output runs/w2a8_bitblas_eval \
  --eval-checkpoint runs/w2a8/best.pt --cuda-w2a8-inference
```

평가에서는 모델, 이미지 크기, 양자화 비트, Cropr, seed와 데이터 분할 설정을 체크포인트에서 자동 복원한다.
`--mode`, `--mixup`, `--epochs` 등 학습 옵션을 다시 입력할 필요가 없다.
저장된 실험 설정이 CLI의 해당 설정보다 우선한다. 장치, 데이터 경로, 출력 경로, 배치 크기,
workers/threads, latency 반복 설정과 `--cuda-w2a8-inference`는 CLI 값을 사용한다.
평가에는 optimizer를 만들지 않으며, 데이터 분할 해시와 가중치의 strict 로딩 검사는 유지한다.
학습 재개(`--resume`)는 기존처럼 학습 설정 일치를 검사한다.
기존 지표는 fake-quant 결과이고 `cuda_w2a8_*`는 변환 모델의 정확도와 latency이다.
`cuda_w2a8_backend`와 `cuda_w2a8_backend_version`에 구현과 버전을 기록한다.
가중치 바이트 수에는 정렬용 패딩도 포함한다.

`cuda_w2a8_state.pt`는 변환 모델의 state dict이다. 같은 원본 QAT 체크포인트로
BitBLAS 모델을 변환한 뒤 불러온다. 이전 직접 구현의 변환 상태는 호환되지 않으므로 원본 QAT에서 다시 변환한다.
BitBLAS 연산자/컴파일 결과는 state dict에 포함되지 않는다. 학습 재개에는 원래 체크포인트를 사용한다.

## FP32와 latency 비교

```bash
python -m scripts.benchmark_w2a8 --checkpoint runs/w2a8/best.pt \
  --device cuda --batch-size 1 --warmup 20 --repeats 100 \
  --output runs/w2a8_bitblas_benchmark.json
```

같은 체크포인트의 `fp32_master`, `fake_quant_w2a8`, `packed_w2a8`를 비교한다.
`packed_w2a8`는 BitBLAS 모델이며 JSON의 `backend`/`backend_version`에 구현과 버전을 기록한다.

합성 입력을 GPU에 올린 뒤 동기화한 forward 시간을 측정한다. 초기 컴파일과 데이터 로딩·전송은 제외한다.
AMP, TF32, CUDA Graph는 사용하지 않는다. CUDA event에도 호스트 호출 사이 대기 시간이 포함될 수 있다.
합성 입력의 logits 차이는 실제 Food-101 정확도 평가를 대신하지 않는다.

## 검증 상태

로컬은 Windows RTX 3060 Laptop이다. Linux 배포판이 없어 실제 BitBLAS 커널과 RTX 4090 성능은
측정하지 못했다. `tests/test_bitblas_w2a8.py`의 adapter 검사는 라이브러리 호출을 명시적인
테스트 대역으로 대체해 정수 누산·affine 보정·패딩·상태 복원·전체 ViT/Cropr 변환을 검사한다.
이 검사를 native 커널 검증으로 해석하면 안 된다. native 항목은 Linux CUDA에서 실제 BitBLAS를 호출한다.

전체 QAT 지점 대조 검사는 Cropr off, all 범위의 LLF off/on, backbone 범위의 LLF off/on에서
각각 198, 352, 338, 198, 198개 지점을 확인한다. 각 scale·zero-point의 보존과 고정 query의
양자화 결과 일치를 검사하며, 변환 후 남은 fake quantizer는 모두 0개이다.
Forward의 Linear/Conv 경로는 50회, attention 경로는 각각 24, 35, 34, 24, 24회 호출된다.
이는 batch/head별 라이브러리 호출 수가 아니라 모델 수준의 행렬곱 경로 호출 수이다.
학습·평가·상태 저장·벤치마크 연결도 adapter/native 항목으로 검사한다.
테스트 대역의 실행 시간은 BitBLAS나 RTX 4090 성능 측정값으로 사용하지 않는다.

참고: [BitBLAS 지원표](https://github.com/microsoft/BitBLAS#support-matrix),
[설치 조건](https://github.com/microsoft/BitBLAS/blob/main/docs/Installation.md),
[Matmul API](https://github.com/microsoft/BitBLAS/blob/main/bitblas/ops/general_matmul/__init__.py).
