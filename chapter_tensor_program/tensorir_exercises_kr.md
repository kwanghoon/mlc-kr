## TensorIR 연습 문제

```{.python .input n=0}
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np
import IPython
```

### 섹션 1: TensorIR 작성 방법

이 섹션에서는 고수준 명령(예: Numpy 또는 Torch)에 따라 TensorIR을 수동으로 작성해 보겠습니다. 먼저 요소별 덧셈 함수의 예를 제공하여 TensorIR 함수를 작성하기 위해 무엇을 해야 하는지 보여드리겠습니다.

#### 예제: 요소별 덧셈

먼저 Numpy를 사용하여 요소별 덧셈 함수를 작성해 보겠습니다.

```{.python .input n=1}
# 데이터 초기화
a = np.arange(16).reshape(4, 4)
b = np.arange(16, 0, -1).reshape(4, 4)
```

```{.python .input n=2}
# numpy 버전
c_np = a + b
c_np
```

TensorIR을 직접 작성하기 전에, 먼저 고수준 계산 추상화(예: `ndarray + ndarray`)를 저수준 Python 구현(요소 접근 및 연산을 포함한 표준 for 루프)으로 변환해야 합니다.

특히, 출력 배열(또는 버퍼)의 초기 값이 항상 `0`인 것은 아닙니다. 우리는 구현에서 이를 작성하거나 초기화해야 하며, 이는 축소 연산자(예: matmul 및 conv)에 중요합니다.

```{.python .input n=3}
# 저수준 numpy 버전
def lnumpy_add(a: np.ndarray, b: np.ndarray, c: np.ndarray):
  for i in range(4):
    for j in range(4):
      c[i, j] = a[i, j] + b[i, j]
c_lnumpy = np.empty((4, 4), dtype=np.int64)
lnumpy_add(a, b, c_lnumpy)
c_lnumpy
```

이제 한 단계 더 나아가 저수준 NumPy 구현을 TensorIR로 변환해 보겠습니다. 그리고 결과를 NumPy에서 나온 것과 비교하겠습니다.

```{.python .input n=4}
# TensorIR 버전
@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add(A: T.Buffer((4, 4), "int64"),
          B: T.Buffer((4, 4), "int64"),
          C: T.Buffer((4, 4), "int64")):
    T.func_attr({"global_symbol": "add"})
    for i, j in T.grid(4, 4):
      with T.block("C"):
        vi = T.axis.spatial(4, i)
        vj = T.axis.spatial(4, j)
        C[vi, vj] = A[vi, vj] + B[vi, vj]

rt_lib = tvm.build(MyAdd, target="llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.empty((4, 4), dtype=np.int64))
rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)
```

여기까지 TensorIR 함수를 완성했습니다. 다음 연습 문제를 완료하는 데 시간을 할애하시기 바랍니다.

#### 연습 문제 1: 브로드캐스트 덧셈

브로드캐스팅을 사용하여 두 배열을 더하는 TensorIR 함수를 작성하세요.

```{.python .input n=5}
# 데이터 초기화
a = np.arange(16).reshape(4, 4)
b = np.arange(4, 0, -1).reshape(4)
```

```{.python .input n=6}
# numpy 버전
c_np = a + b
c_np
```

다음 모듈 `MyAdd`를 완성하고 코드를 실행하여 구현을 확인하세요.

```python
@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add():
    T.func_attr({"global_symbol": "add", "tir.noalias": True})
    # TODO
    ...

rt_lib = tvm.build(MyAdd, target="llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.empty((4, 4), dtype=np.int64))
rt_lib["add"](a_tvm, b_tvm, c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)
```

#### 연습 문제 2: 2D 합성곱

이제 좀 더 도전적인 것을 시도해 봅시다: 이미지 처리에서 일반적인 연산인 2D 합성곱입니다.

다음은 NCHW 레이아웃을 사용한 합성곱의 수학적 정의입니다:

$$Conv[b, k, i, j] =
    \sum_{di, dj, q} A[b, q, strides * i + di, strides * j + dj] * W[k, q, di, dj]$$
여기서 `A`는 입력 텐서, `W`는 가중치 텐서, `b`는 배치 인덱스, `k`는 출력 채널, `i`와 `j`는 이미지 높이와 너비의 인덱스, `di`와 `dj`는 가중치의 인덱스, `q`는 입력 채널, `strides`는 필터 윈도우의 스트라이드입니다.

이 연습에서는 `stride=1, padding=0`인 작고 간단한 경우를 선택합니다.

```{.python .input n=7}
N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H - K + 1, W - K + 1
data = np.arange(N*CI*H*W).reshape(N, CI, H, W)
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K)
```

```{.python .input n=8}
# torch 버전
import torch
data_torch = torch.Tensor(data)
weight_torch = torch.Tensor(weight)
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
conv_torch = conv_torch.numpy().astype(np.int64)
conv_torch
```

다음 모듈 `MyConv`를 완성하고 코드를 실행하여 구현을 확인하세요.

```python
@tvm.script.ir_module
class MyConv:
  @T.prim_func
  def conv():
    T.func_attr({"global_symbol": "conv", "tir.noalias": True})
    # TODO
    ...

rt_lib = tvm.build(MyConv, target="llvm")
data_tvm = tvm.nd.array(data)
weight_tvm = tvm.nd.array(weight)
conv_tvm = tvm.nd.array(np.empty((N, CO, OUT_H, OUT_W), dtype=np.int64))
rt_lib["conv"](data_tvm, weight_tvm, conv_tvm)
np.testing.assert_allclose(conv_tvm.numpy(), conv_torch, rtol=1e-5)
```

### 섹션 2: TensorIR 변환 방법

강의에서 우리는 TensorIR이 프로그래밍 언어일 뿐만 아니라 프로그램 변환을 위한 추상화이기도 하다는 것을 배웠습니다. 이 섹션에서는 프로그램을 변환해 보겠습니다. 우리는 연구에서 `bmm_relu`(`batched_matmul_relu`)를 사용하는데, 이는 트랜스포머와 같은 모델에서 흔히 나타나는 연산의 변형입니다.

#### 병렬화, 벡터화 및 언롤
먼저 몇 가지 새로운 기본 요소인 `parallel`, `vectorize`, `unroll`을 소개합니다. 이 세 가지 기본 요소는 루프에서 작동하여 이 루프가 어떻게 실행되는지를 나타냅니다. 다음은 예제입니다:

```{.python .input n=9}
@tvm.script.ir_module
class MyAdd:
  @T.prim_func
  def add(A: T.Buffer((4, 4), "int64"),
          B: T.Buffer((4, 4), "int64"),
          C: T.Buffer((4, 4), "int64")):
    T.func_attr({"global_symbol": "add"})
    for i, j in T.grid(4, 4):
      with T.block("C"):
        vi = T.axis.spatial(4, i)
        vj = T.axis.spatial(4, j)
        C[vi, vj] = A[vi, vj] + B[vi, vj]

sch = tvm.tir.Schedule(MyAdd)
block = sch.get_block("C", func_name="add")
i, j = sch.get_loops(block)
i0, i1 = sch.split(i, factors=[2, 2])
sch.parallel(i0)
sch.unroll(i1)
sch.vectorize(j)
IPython.display.Code(sch.mod.script(), language="python")
```

#### 연습 문제 3: 배치 행렬 곱셈 프로그램 변환
이제 `bmm_relu` 연습으로 돌아가겠습니다. 먼저 `bmm`의 정의를 살펴보겠습니다:

- $Y_{n, i, j} = \sum_k A_{n, i, k} \times B_{n, k, j}$
- $C_{n, i, j} = \mathbb{relu}(Y_{n,i,j}) = \mathbb{max}(Y_{n, i, j}, 0)$

이제 `bmm_relu`에 대한 TensorIR을 작성할 차례입니다. 힌트로 lnumpy 함수를 제공합니다:

```{.python .input n=10}
def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((16, 128, 128), dtype="float32")
    for n in range(16):
        for i in range(128):
            for j in range(128):
                for k in range(128):
                    if k == 0:
                        Y[n, i, j] = 0
                    Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
    for n in range(16):
        for i in range(128):
            for j in range(128):
                C[n, i, j] = max(Y[n, i, j], 0)
```

```python
@tvm.script.ir_module
class MyBmmRelu:
  @T.prim_func
  def bmm_relu():
    T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
    # TODO
    ...

sch = tvm.tir.Schedule(MyBmmRelu)
IPython.display.Code(sch.mod.script(), language="python")
# 또한 결과를 검증하세요
```

이 연습에서는 원본 프로그램을 특정 타겟으로 변환하는 데 집중하겠습니다. 타겟 프로그램이 하드웨어마다 다를 수 있으므로 최선이 아닐 수도 있습니다. 하지만 이 연습의 목적은 학생들이 프로그램을 원하는 것으로 변환하는 방법을 이해하도록 하는 것입니다. 다음은 타겟 프로그램입니다:

```{.python .input n=11}
@tvm.script.ir_module
class TargetModule:
    @T.prim_func
    def bmm_relu(A: T.Buffer((16, 128, 128), "float32"), B: T.Buffer((16, 128, 128), "float32"), C: T.Buffer((16, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
        Y = T.alloc_buffer([16, 128, 128], dtype="float32")
        for i0 in T.parallel(16):
            for i1, i2_0 in T.grid(128, 16):
                for ax0_init in T.vectorized(8):
                    with T.block("Y_init"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + ax0_init)
                        Y[n, i, j] = T.float32(0)
                for ax1_0 in T.serial(32):
                    for ax1_1 in T.unroll(4):
                        for ax0 in T.serial(8):
                            with T.block("Y_update"):
                                n, i = T.axis.remap("SS", [i0, i1])
                                j = T.axis.spatial(128, i2_0 * 8 + ax0)
                                k = T.axis.reduce(128, ax1_0 * 4 + ax1_1)
                                Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
                for i2_1 in T.vectorized(8):
                    with T.block("C"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + i2_1)
                        C[n, i, j] = T.max(Y[n, i, j], T.float32(0))
```

여러분의 작업은 원본 프로그램을 타겟 프로그램으로 변환하는 것입니다.

```python
sch = tvm.tir.Schedule(MyBmmRelu)
# TODO: 변환
# 힌트: 다음을 사용할 수 있습니다
# `IPython.display.Code(sch.mod.script(), language="python")`
# 또는 `print(sch.mod.script())`
# 변환 중 언제든지 현재 프로그램을 보여줄 수 있습니다.

# 단계 1. 블록 가져오기
Y = sch.get_block("Y", func_name="bmm_relu")
...

# 단계 2. 루프 가져오기
b, i, j, k = sch.get_loops(Y)
...

# 단계 3. 루프 구성
k0, k1 = sch.split(k, ...)
sch.reorder(...)
sch.compute_at/reverse_compute_at(...)
...

# 단계 4. 축소 분해
Y_init = sch.decompose_reduction(Y, ...)
...

# 단계 5. 벡터화 / 병렬화 / 언롤
sch.vectorize(...)
sch.parallel(...)
sch.unroll(...)
...

IPython.display.Code(sch.mod.script(), language="python")
```

**선택 사항** 변환된 프로그램이 주어진 타겟과 정확히 동일한지 확인하려면 `assert_structural_equal`을 사용할 수 있습니다. 이 단계는 이 연습에서 선택 사항입니다. 프로그램을 타겟 **방향으로** 변환하고 성능 향상을 얻으면 충분합니다.

```python
tvm.ir.assert_structural_equal(sch.mod, TargetModule)
print("Pass")
```

#### 빌드 및 평가

마지막으로 변환된 프로그램의 성능을 평가할 수 있습니다.

```python
before_rt_lib = tvm.build(MyBmmRelu, target="llvm")
after_rt_lib = tvm.build(sch.mod, target="llvm")
a_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
b_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
c_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
after_rt_lib["bmm_relu"](a_tvm, b_tvm, c_tvm)
before_timer = before_rt_lib.time_evaluator("bmm_relu", tvm.cpu())
print("Before transformation:")
print(before_timer(a_tvm, b_tvm, c_tvm))

f_timer = after_rt_lib.time_evaluator("bmm_relu", tvm.cpu())
print("After transformation:")
print(f_timer(a_tvm, b_tvm, c_tvm))
```
