## TensorIR: 텐서 프로그램 추상화 사례 연구

### 패키지 설치

이 강좌를 위해 오픈 소스 머신러닝 컴파일 프레임워크인 TVM의 진행 중인 개발 버전을 사용할 것입니다. MLC 강좌를 위한 패키지 버전을 설치하기 위해 다음 명령어를 제공합니다.

```bash
python3 -m  pip install mlc-ai-nightly -f https://mlc.ai/wheels
```

### 서론

![](../img/tensor_func_linear_relu.png)

오늘 강의를 시작하기 전에, MLC 프로세스의 핵심 원리를 되짚어 봅시다. 대부분의 MLC 프로세스는 텐서 함수 간의 변환으로 볼 수 있습니다. 우리가 다음 내용에서 답하고자 하는 주요 질문은:

- 텐서 함수를 표현하기 위한 가능한 추상화는 무엇인가.
- 텐서 함수 간의 가능한 변환은 무엇인가.

오늘은 원시 텐서 함수에 초점을 맞추어 그 일부를 다룰 것입니다.

### 하나의 텐서 프로그램 추상화 학습하기 -- TensorIR

우리는 원시 텐서 함수를 살펴보고 텐서 프로그램 추상화의 고수준 아이디어에 대해 논의했습니다.

이제 TensorIR이라는 텐서 프로그램 추상화의 특정 인스턴스를 배울 준비가 되었습니다. TensorIR은 표준 머신러닝 컴파일 프레임워크 중 하나인 Apache TVM의 텐서 프로그램 추상화입니다.

```{.python .input n=0}
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np
```

텐서 프로그램 추상화의 주요 목적은 루프와 스레딩, 특수 하드웨어 명령어 사용, 메모리 접근과 같은 해당 하드웨어 가속 선택을 표현하는 것입니다.

설명을 돕기 위해 다음 텐서 계산 시퀀스를 동기 부여 예제로 사용하겠습니다.

구체적으로, 두 개의 $128 \times 128$ 행렬 A와 B에 대해 다음 두 단계의 텐서 계산을 수행합니다.

- $Y_{i, j} = \sum_k A_{i, k} \times B_{k, j}$
- $C_{i, j} = \mathbb{relu}(Y_{i, j}) = \mathbb{max}(Y_{i, j}, 0)$

위의 계산은 신경망에서 흔히 볼 수 있는 전형적인 원시 텐서 함수 -- relu 활성화를 가진 선형 레이어를 닮았습니다. 시작하기 위해, NumPy의 배열 계산을 사용하여 두 연산을 다음과 같이 구현할 수 있습니다.

```{.python .input n=1}
dtype = "float32"
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)
# a @ b is equivalent to np.matmul(a, b)
c_mm_relu = np.maximum(a_np @ b_np, 0)
```

내부적으로 NumPy는 이러한 계산을 실행하기 위해 라이브러리(예: OpenBLAS)와 저수준 C 언어로 작성된 자체 구현을 호출합니다.

텐서 프로그램 추상화 관점에서, 우리는 이러한 배열 계산의 **내부** 세부 사항을 살펴보고 싶습니다. 구체적으로, 우리는 다음을 질문하고 싶습니다: 해당 계산을 구현하는 가능한 방법은 무엇인가?

내부 세부 사항을 설명하기 위해, 우리는 NumPy API의 제한된 부분 집합으로 예제를 작성할 것인데 -- 이를 다음 관례를 사용하는 **저수준 numpy**라고 부릅니다:

- 가능한 루프 계산을 보여주기 위해 필요할 때 배열 함수 대신 루프를 사용합니다.
- 가능한 경우, 항상 numpy.empty를 통해 배열을 명시적으로 할당하고 전달합니다.

이것은 일반적으로 NumPy 프로그램을 작성하는 방식이 아닙니다. 그러나 내부적으로 일어나는 일과 매우 유사합니다 -- 대부분의 실제 배포 솔루션은 계산과 별도로 할당을 처리합니다. 특정 라이브러리는 다양한 형태의 루프와 산술 계산을 사용하여 계산을 수행합니다. 물론 주로 `C`와 같은 저수준 언어를 사용하여 구현됩니다.

```{.python .input n=2}
def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0)
```

위 프로그램은 `mm_relu` 연산을 구현하는 한 가지 방법입니다. 프로그램은 두 단계로 구성됩니다: 먼저 중간 저장공간 $Y$를 할당하고 거기에 행렬 곱셈의 결과를 저장합니다. 그런 다음 두 번째 for 루프 시퀀스에서 relu를 계산합니다. 여러분이 알아차릴 수 있는 한 가지는 이것이 `mm_relu`를 구현하는 유일한 방법이 확실히 아니라는 것입니다. 아마도 여러분이 처음에 떠올릴 방법도 아닐 것입니다.

그럼에도 불구하고, 이것은 `mm_relu`를 구현하는 한 가지 방법이며, 배열 계산을 사용한 원래 결과와 비교하여 코드의 정확성을 검증할 수 있습니다. 이 튜토리얼의 후반부에서 다른 가능한 방법들을 다시 살펴보겠습니다.

```{.python .input n=3}
c_np = np.empty((128, 128), dtype=dtype)
lnumpy_mm_relu(a_np, b_np, c_np)
np.testing.assert_allclose(c_mm_relu, c_np, rtol=1e-5)
```

위의 예제 코드는 `mm_relu`의 **내부** 구현을 어떻게 가져올 수 있는지를 보여줍니다. 물론 Python 인터프리터 때문에 코드 자체는 훨씬 느리게 실행될 것입니다. 그럼에도 불구하고, 예제 numpy 코드는 이러한 계산의 실제 구현에서 사용할 모든 가능한 요소를 포함하고 있습니다.

- 다차원 버퍼(배열).
- 배열 차원에 대한 루프.
- 루프 하에서 실행되는 계산 구문.

저수준 NumPy 예제를 염두에 두고, 이제 TensorIR을 소개할 준비가 되었습니다. 아래 코드 블록은 `mm_relu`의 TensorIR 구현을 보여줍니다. 이 특정 코드는 Python AST에 내장된 도메인별 방언인 TVMScript라는 언어로 구현되었습니다.

```{.python .input n=4}
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"),
                B: T.Buffer((128, 128), "float32"),
                C: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                vk = T.axis.reduce(128, k)
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                vj = T.axis.spatial(128, j)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))
```

numpy 코드와 TensorIR 코드를 나란히 보고 해당 요소들을 확인할 수 있으면 도움이 되며, 각 요소를 자세히 살펴보겠습니다.

![](../img/tensor_func_and_numpy.png)

먼저 numpy와 TensorIR 측면 사이에 직접적인 대응 관계가 있는 요소들을 검토하는 것부터 시작하겠습니다. 그런 다음 numpy 프로그램의 일부가 아닌 추가 요소들을 다시 검토하겠습니다.

#### 함수 매개변수와 버퍼
먼저 함수 매개변수를 살펴보겠습니다. 함수 매개변수는 numpy 함수의 동일한 매개변수 세트에 해당합니다.

```python
# TensorIR
def mm_relu(A: T.Buffer[(128, 128), "float32"],
            B: T.Buffer[(128, 128), "float32"],
            C: T.Buffer[(128, 128), "float32"]):
    ...
# numpy
def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    ...
```

여기서 A, B, C는 `T.Buffer`라는 타입을 취하며, shape 인자로 `(128, 128)`을, 데이터 타입으로 `float32`를 가집니다. 이 추가 정보는 MLC 프로세스가 모양과 데이터 타입에 특화된 코드를 생성하는 데 도움이 됩니다.

마찬가지로 TensorIR도 중간 결과 할당에서 버퍼 타입을 사용합니다.

```python
# TensorIR
Y = T.alloc_buffer((128, 128), dtype="float32")
# numpy
Y = np.empty((128, 128), dtype="float32")
```

#### For 루프 반복

루프 반복도 직접적인 대응 관계가 있습니다. `T.grid`는 TensorIR에서 여러 중첩 반복자를 작성하기 위한 문법적 설탕입니다.


```python
# TensorIR
for i, j, k in T.grid(128, 128, 128):

# numpy
for i in range(128):
    for j in range(128):
        for k in range(128):
```

#### 계산 블록

주요 차이점 중 하나는 계산 구문에서 비롯됩니다. TensorIR은 `T.block`이라는 추가 구조를 포함합니다.

```python
# TensorIR
with T.block("Y"):
    vi = T.axis.spatial(128, i)
    vj = T.axis.spatial(128, j)
    vk = T.axis.reduce(128, k)
    with T.init():
        Y[vi, vj] = T.float32(0)
    Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

# 해당 numpy 코드
vi, vj, vk = i, j, k
if vk == 0:
    Y[vi, vj] = 0
Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

```

**블록**은 TensorIR의 기본 계산 단위입니다. 특히, 블록은 일반 NumPy 코드에 비해 모7 가지 추가 정보를 포함합니다. 블록은 블록 축(`vi, vj, vk`) 집합과 그 주위에 정의된 계산을 포함합니다.

```python
vi = T.axis.spatial(128, i)
vj = T.axis.spatial(128, j)
vk = T.axis.reduce(128, k)
```

The above three lines declare the **key properties** about block axes in the following syntax.
```
[block_axis] = T.axis.[axis_type]([axis_range], [mapped_value])
```

이 세 줄은 다음 정보를 포함합니다:

- vi, vj, vk가 무엇에 바인드되어야 하는지 정의합니다(이 경우 i, j, k).
- vi, vj, vk가 가져야 할 원래 범위를 선언합니다(`T.axis.spatial(128, i)`의 `128`).
- 반복자의 속성을 선언합니다(`spatial`, `reduce`).

이러한 속성들을 하나씩 살펴보겠습니다. 먼저 바인딩 관계 측면에서, `vi = T.axis.spatial(128, i)`는 사실상 `vi = i`를 의미합니다. `[axis_range]` 값은 `[block_axis]`의 예상 범위를 제공합니다. 예를 들어, `vi = T.axis.spatial(128, i)`의 `128`은 `vi`가 `range(0, 128)` 안에 있어야 한다는 것을 나타냅니다.

#### 블록 축 속성

이제 블록 축 속성을 자세히 살펴보겠습니다. 이러한 축 속성은 수행되는 계산에 대한 축의 관계를 표시합니다.
아래 그림은 블록(반복) 축과 블록 Y의 읽기 쓰기 관계를 요약합니다. 엄밀히 말하면 블록은 `Y`에 (축소) 업데이트를 수행하고 있지만, 다른 블록의 `Y` 값이 필요하지 않으므로 지금은 이를 쓰기로 표시합니다.

![](../img/tensor_ir_block_axis.png)

우리 예제에서 블록 Y는 `A[vi, vk]`와 `B[vk, vj]`에서 값을 읽고 모든 가능한 `vk`에 대해 합을 수행하여 결과 `Y[vi, vj]`를 계산합니다. 이 특정 예제에서 `vi`, `vj`를 `(0, 1)`로 고정하고 `vk in range(0, 128)`에 대해 블록을 실행하면, Y의 다른 위치(다른 vi, vj 값을 가진)와 독립적으로 `C[0, 1]`을 효과적으로 계산할 수 있습니다.

특히, vi와 vj의 고정된 값에 대해, 계산 블록은 `Y`의 다른 위치(다른 `vi, vj` 값을 가진)와 독립적인 Y의 공간적 위치(`Y[vi, vj]`)에서 점 값을 생성합니다. `vi`, `vj`를 **공간 축(spatial axes)**이라고 부를 수 있는데, 이들은 블록이 쓰기를 하는 버퍼의 공간적 영역의 시작에 직접 대응하기 때문입니다. 축소에 관련된 축(`vk`)은 **축소 축(reduce axes)**이라고 합니다.

#### 블록의 추가 정보가 필요한 이유

한 가지 중요한 관찰은 추가 정보(블록 축 범위와 그 속성)가 블록이 외부 루프 중첩 `i`, `j`, `k`로부터 독립적으로 수행해야 하는 반복과 관련하여 블록을 **자체 포함적(self-contained)**으로 만든다는 것입니다.

블록 축 정보는 또한 계산을 수행하는 데 사용되는 외부 루프의 정확성을 검증하는 데 도움이 되는 추가 속성을 제공합니다. 예를 들어, 아래 코드 블록은 루프가 크기 `128`의 반복자를 기대하지만 크기 `127`의 for 루프에만 바인드했기 때문에 오류가 발생합니다.


```python
# wrong program due to loop and block iteration mismatch
for i in range(127):
    with T.block("C"):
        vi = T.axis.spatial(128, i)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
        error here due to iterator size mismatch
        ...
```

이 추가 정보는 또한 머신러닝 컴파일 분석에도 도움이 됩니다. 예를 들어, 공간 축에 대해서는 항상 병렬화할 수 있지만, 축소 축에 대한 병렬화는 특정 전략이 필요합니다.

#### 블록 축 바인딩을 위한 문법적 설탕

각 블록 축이 외부 루프 반복자에 직접 매핑되는 상황에서, `T.axis.remap`을 사용하여 한 줄로 블록 축을 선언할 수 있습니다.

```python
# SSR means the properties of each axes are "spatial", "spatial", "reduce"
vi, vj, vk = T.axis.remap("SSR", [i, j, k])
```
is equivalent to
```python
vi = T.axis.spatial(range_of_i, i)
vj = T.axis.spatial(range_of_j, j)
vk = T.axis.reduce(range_of_k, k)
```

So we can also write the programs as follows.

```{.python .input n=5}
@tvm.script.ir_module
class MyModuleWithAxisRemapSugar:
    @T.prim_func
    def mm_relu(A: T.Buffer((128, 128), "float32"),
                B: T.Buffer((128, 128), "float32"),
                C: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(128, 128):
            with T.block("C"):
                vi, vj = T.axis.remap("SS", [i, j])
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))
```

#### 함수 속성과 데코레이터

지금까지 TensorIR의 대부분의 요소를 다루었습니다. 이 부분에서는 스크립트의 나머지 요소들을 살펴보겠습니다.

함수 속성 정보는 함수에 대한 추가 정보를 포함합니다.

```python
T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
```

여기서 `global_symbol`은 함수의 이름에 해당하고, `tir.noalias`는 모든 버퍼 메모리 영역이 겹치지 않음을 나타내는 속성입니다. 이러한 속성들은 고수준 개념의 전반적인 이해에 영향을 미치지 않으므로 지금은 안전하게 건너뛰어도 됩니다.

두 개의 데코레이터 `@tvm.script.ir_module`과 `@T.prim_func`는 해당 부분의 타입을 나타내는 데 사용됩니다.

`@tvm.script.ir_module`은 MyModule이 `IRModule`임을 나타냅니다. IRModule은 머신러닝 컴파일에서 텐서 함수의 모음을 보유하는 컨테이너 객체입니다.


```{.python .input n=6}
type(MyModule)
```

```{.python .input n=7}
type(MyModule["mm_relu"])
```

지금까지는 단일 텐서 함수를 포함하는 IRModule만 보았습니다. MLC 프로세스의 IRModule은 여러 텐서 함수를 포함할 수 있습니다. 다음 코드 블록은 두 개의 함수를 가진 IRModule의 예를 보여줍니다.

```{.python .input n=8}
@tvm.script.ir_module
class MyModuleWithTwoFunctions:
    @T.prim_func
    def mm(A: T.Buffer((128, 128), "float32"),
           B: T.Buffer((128, 128), "float32"),
           Y: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "mm", "tir.noalias": True})
        for i, j, k in T.grid(128, 128, 128):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]

    @T.prim_func
    def relu(A: T.Buffer((128, 128), "float32"),
             B: T.Buffer((128, 128), "float32")):
        T.func_attr({"global_symbol": "relu", "tir.noalias": True})
        for i, j in T.grid(128, 128):
            with T.block("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], T.float32(0))
```

#### 섹션 체크포인트

지금까지 TensorIR 프로그램의 한 가지 예시를 살펴보고 다음을 포함한 대부분의 요소를 다루었습니다:

- 매개변수와 중간 임시 메모리의 버퍼 선언.
- For 루프 반복.
- **블록**과 블록 축 속성.

이 섹션에서는 MLC에서 가장 일반적인 요소를 다루는 TensorIR의 한 가지 예시를 살펴보았습니다.

TensorIR은 이 섹션에서 다룬 것보다 더 많은 요소를 포함하지만, 이 섹션은 MLC 여정을 시작하는 데 필요한 대부분의 핵심 부분을 다룹니다. 나중 장에서 새로운 요소를 만나면 다룰 것입니다.

### 변환

이전 섹션에서는 TensorIR과 그 핵심 요소들에 대해 배웠습니다. 이제 모든 MLC 플로우의 주요 구성 요소인 원시 텐서 함수의 변환에 대해 알아보겠습니다.

이전 섹션에서 저수준 numpy를 사용하여 `mm_relu`를 작성하는 방법의 예를 제시했습니다. 실제로 동일한 기능을 구현하는 여러 방법이 있을 수 있으며, 각 구현은 다른 성능을 초래할 수 있습니다.

성능 차이의 이유와 이러한 변형들을 활용하는 방법은 향후 강의에서 논의할 것입니다. 이 강의에서는 변환을 사용하여 다른 구현 변형을 얻는 능력에 집중하겠습니다.

```{.python .input n=9}
def lnumpy_mm_relu_v2(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j0 in range(32):
            for k in range(128):
                for j1 in range(4):
                    j = j0 * 4 + j1
                    if k == 0:
                        Y[i, j] = 0
                    Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0)

c_np = np.empty((128, 128), dtype=dtype)
lnumpy_mm_relu_v2(a_np, b_np, c_np)
np.testing.assert_allclose(c_mm_relu, c_np, rtol=1e-5)
```

위 코드 블록은 `mm_relu`의 약간 다른 변형을 보여줍니다. 원래 프로그램과의 관계를 보면:

- `j` 루프를 두 개의 루프 `j0`과 `j1`로 교체합니다.
- 반복 순서가 약간 변경됩니다.

`lnumpy_mm_relu_v2`를 얻기 위해서는 새로운 함수로 다시 작성하거나(또는 수동으로 복사-붙여넣기 및 편집) 해야 합니다. TensorIR은 이를 프로그래밍 방식으로 수행할 수 있도록 하는 Schedule이라는 유틸리티를 도입합니다.

상기시키기 위해, 현재 MyModule 콘텐츠를 다시 살펴보겠습니다.

```{.python .input n=10}
import IPython
IPython.display.Code(MyModule.script(), language="python")
```

이제 코드 변환을 시도할 준비가 되었습니다. 먼저 주어진 MyModule을 입력으로 하여 `Schedule` 헬퍼 클래스를 생성합니다.

```{.python .input n=11}
sch = tvm.tir.Schedule(MyModuleWithAxisRemapSugar)
```

그런 다음 블록 Y와 해당 루프에 대한 참조를 얻기 위해 다음 작업을 수행합니다.

```{.python .input n=12}
block_Y = sch.get_block("Y", func_name="mm_relu")
i, j, k = sch.get_loops(block_Y)
```

이제 변환을 수행할 준비가 되었습니다. 수행할 첫 번째 변환은 루프 `j`를 두 개의 루프로 분할하는 것으로, 내부 루프의 길이는 `4`입니다. 변환은 절차적이므로 블록을 실수로 두 번 실행하면 변수 `j`가 더 이상 존재하지 않는다는 오류가 발생합니다. 그런 일이 발생하면 처음부터(`sch`가 생성되는 곳) 다시 실행할 수 있습니다.

```{.python .input n=13}
j0, j1 = sch.split(j, factors=[None, 4])
```

`sch.mod`에 저장된 변환 결과를 볼 수 있습니다.

```{.python .input n=14}
IPython.display.Code(sch.mod.script(), language="python")
```

첫 번째 변환 단계 후에 해당 범위가 32와 4인 두 개의 추가 루프 `j_0`과 `j_1`을 생성했습니다. 다음 단계는 두 루프를 재정렬하는 것입니다.

```{.python .input n=15}
sch.reorder(j0, k, j1)
IPython.display.Code(sch.mod.script(), language="python")
```

재정렬 후의 코드는 이제 `lnumpy_mm_relu_v2`와 매우 유사해졌습니다.

#### 다른 변형 얻기

이 섹션에서는 또 다른 변형을 얻기 위해 두 단계의 변환을 더 진행하겠습니다. 먼저 `reverse_compute_at`이라는 기본 요소를 사용하여 블록 C를 `Y`의 내부 루프로 이동시킵니다.

```{.python .input n=16}
block_C = sch.get_block("C", "mm_relu")
sch.reverse_compute_at(block_C, j0)
IPython.display.Code(sch.mod.script(), language="python")
```

지금까지 축소 초기화와 업데이트 단계를 단일 블록 본문에 함께 유지해 왔습니다. 이 결합된 형태는 루프 변환에 편의함을 제공합니다(초기화와 업데이트의 외부 루프 i,j가 일반적으로 서로 동기화를 유지해야 하기 때문).

루프 변환 후에는 `Y`의 요소 초기화를 축소 업데이트와 분리할 수 있습니다. `decompose_reduction` 기본 요소를 통해 이를 수행할 수 있습니다. (참고: 이것은 향후 컴파일 중에 tvm에 의해 암묵적으로 수행되므로, 이 단계는 주로 명시적으로 만들고 최종 효과를 보기 위한 것입니다).

```
sch.decompose_reduction(block_Y, k)
IPython.display.Code(sch.mod.script(), language="python")
```

최종 변환된 코드는 다음 저수준 NumPy 코드와 유사합니다.

```{.python .input n=17}
def lnumpy_mm_relu_v3(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j0 in range(32):
            # Y_init
            for j1 in range(4):
                j = j0 * 4 + j1
                Y[i, j] = 0
            # Y_update
            for k in range(128):
                for j1 in range(4):
                    j = j0 * 4 + j1
                    Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
            # C
            for j1 in range(4):
                j = j0 * 4 + j1
                C[i, j] = max(Y[i, j], 0)

c_np = np.empty((128, 128), dtype=dtype)
lnumpy_mm_relu_v3(a_np, b_np, c_np)
np.testing.assert_allclose(c_mm_relu, c_np, rtol=1e-5)
```

#### 섹션 요약 및 토론

이 섹션의 주요 요점은 점진적 코드 변환의 패러다임에 익숙해지는 것입니다. 우리의 특정 예제에서는 `tir.Schedule`을 보조 헬퍼 객체로 사용했습니다.

중요한 것은, 동일한 프로그램의 다른 변형들(`lnumpy_mm_relu`, `lnumpy_mm_relu_v2`, `lnumpy_mm_relu_v3`)을 다시 생성할 필요를 피했다는 것입니다. 블록의 추가 정보(축 정보)가 내부에서 이러한 변환을 수행할 수 있는 이유입니다.

### 빌드 및 실행

지금까지는 변환된 결과의 스크립트 출력만 살펴보았습니다. IRModule에서 얻은 프로그램을 실행할 수도 있습니다.

먼저, 빌드 함수를 호출하여 IRModule을 실행 가능한 함수의 모음을 나타내는 `runtime.Module`로 변환합니다. 여기서 target은 배포 환경에 대한 자세한 정보를 지정합니다. 이 특정 경우에는 네이티브 CPU 플랫폼으로 컴파일하는 데 도움이 되는 `llvm`을 사용할 것입니다.

다른 플랫폼(예: 안드로이드 폰)이나 특수 명령어가 있는 플랫폼(Intel Skylake)을 대상으로 할 때는 그에 따라 target을 조정해야 합니다. 이러한 환경에 배포하기 시작할 때 다른 target 선택에 대해 논의할 것입니다.

```{.python .input n=18}
rt_lib = tvm.build(MyModule, target="llvm")
```

그런 다음 입력과 출력을 보유하는 데 사용되는 세 개의 tvm ndarray를 생성합니다.

```{.python .input n=19}
a_nd = tvm.nd.array(a_np)
b_nd = tvm.nd.array(b_np)
c_nd = tvm.nd.empty((128, 128), dtype="float32")
type(c_nd)
```

마지막으로 rt_lib에서 실행 가능한 함수를 가져와 세 개의 배열 인자를 전달하여 실행할 수 있습니다. 코드 차이를 확인하기 위해 검증을 더 실행할 수 있습니다.

```{.python .input n=20}
func_mm_relu = rt_lib["mm_relu"]
func_mm_relu(a_nd, b_nd, c_nd)

np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)
```

원래 MyModule을 빌드하고 실행했습니다. 변환된 프로그램도 빌드할 수 있습니다.

```{.python .input n=21}
rt_lib_after = tvm.build(sch.mod, target="llvm")
rt_lib_after["mm_relu"](a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)
```

마지막으로 두 개의 시간 차이를 비교할 수 있습니다. `time_evaluator`는 다른 생성된 함수들의 실행 성능을 비교하는 데 사용할 수 있는 헬퍼 벤치마킹 함수입니다.

```{.python .input n=22}
f_timer_before = rt_lib.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of MyModule %g sec" % f_timer_before(a_nd, b_nd, c_nd).mean)
f_timer_after = rt_lib_after.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of transformed sch.mod %g sec" % f_timer_after(a_nd, b_nd, c_nd).mean)
```

두 코드 간의 실행 시간 차이를 보는 것은 흥미롭습니다. 성능에 영향을 미치는 가능한 요인에 대한 빠른 분석을 해보겠습니다. 먼저 코드의 두 변형을 다시 상기시켜 보겠습니다.

```{.python .input n=23}
import IPython
IPython.display.Code(MyModule.script(), language="python")
```

```{.python .input n=24}
IPython.display.Code(sch.mod.script(), language="python")
```

![](../img/cpu_arch.png)

다른 루프 변형이 다른 성능을 초래하는 이유를 이해하려면, `A`와 `B`의 메모리 조각에 접근하는 속도가 균일하게 빠른 것은 아니라는 사실을 검토해야 합니다. 현대 CPU는 여러 레벨의 캐시를 가지고 있으며, CPU가 접근하기 전에 데이터를 캐시로 가져와야 합니다.

중요한 것은, 이미 캐시에 있는 데이터에 접근하는 것이 훨씬 빠르다는 것입니다. CPU가 취하는 한 가지 전략은 서로 가까운 데이터를 가져오는 것입니다. 메모리에서 한 요소를 읽을 때, 가까운 요소들(공식적으로 캐시 라인으로 알려진)을 캐시로 가져오려고 시도합니다. 그래서 다음 요소를 읽을 때 이미 캐시에 있습니다. 그 결과, 연속적인 메모리 접근을 가진 코드는 메모리의 다른 부분에 무작위로 접근하는 코드보다 일반적으로 빠릅니다.


![](../img/tensor_func_loop_order.png)

이제 위의 반복 시각화를 보고 무슨 일이 일어나는지 분석해 보겠습니다.
이 분석에서는 두 개의 가장 안쪽 루프인 `k`와 `j1`에 집중하겠습니다. 강조 표시된 덜개는 `k`의 특정 인스턴스에 대해 `j1`을 반복할 때 반복이 접촉하는 `Y`, `A`, `B`의 해당 영역을 보여줍니다.

`j1` 반복이 `B`의 요소들에 대한 **연속적인 접근**을 생성한다는 것을 알 수 있습니다. 구체적으로, `j1=0`일 때와 `j1=1`일 때 읽는 값들이 서로 인접해 있다는 의미입니다. 이는 더 나은 캐시 접근 동작을 가능하게 합니다. 또한 C의 계산을 `Y`에 더 가까이 가져와서 더 나은 캐싱 동작을 가능하게 합니다.

현재 예제는 주로 코드의 다른 변형이 다른 성능으로 이어질 수 있음을 보여주기 위한 것입니다. 더 많은 변환 단계가 훨씬 더 나은 성능을 얻는 데 도움이 될 수 있으며, 이는 향후 장에서 다룰 것입니다. 이 연습의 주요 목표는 먼저 프로그램 변환 도구를 얻고 변환을 통해 가능한 것에 대한 첫 경험을 하는 것입니다.

#### 연습 문제
연습으로, 다른 `j_factor` 선택을 시도하고 그것이 코드의 성능에 어떻게 영향을 미치는지 확인하세요.

```{.python .input n=25}
def transform(mod, jfactor):
    sch = tvm.tir.Schedule(mod)
    block_Y = sch.get_block("Y", func_name="mm_relu")
    i, j, k = sch.get_loops(block_Y)
    j0, j1 = sch.split(j, factors=[None, jfactor])
    sch.reorder(j0, k, j1)
    block_C = sch.get_block("C", "mm_relu")
    sch.reverse_compute_at(block_C, j0)
    return sch.mod

mod_transformed = transform(MyModule, jfactor=8)

rt_lib_transformed = tvm.build(mod_transformed, "llvm")
f_timer_transformed = rt_lib_transformed.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of transformed mod_transformed %g sec" % f_timer_transformed(a_nd, b_nd, c_nd).mean)
# display the code below
IPython.display.Code(mod_transformed.script(), language="python")
```

### TensorIR을 생성하고 상호작용하는 방법

이전 섹션에서는 TensorIR 추상화와 변환 방법에 대해 배웠습니다. TensorIR은 코드 변환을 분석하고 수행하는 데 도움이 되는 블록이라는 추가 구조를 가지고 있습니다. 우리가 물어볼 수 있는 자연스러운 질문은: TensorIR 함수를 생성하고 상호작용하는 일반적인 방법은 무엇인가요?

#### TVMScript를 통한 TensorIR 생성

TensorIR 함수를 얻는 첫 번째 방법은 TVMScript로 함수를 직접 작성하는 것이며, 이는 이전 섹션에서 사용한 접근 방식이기도 합니다. TVMScript는 필요할 때 특정 정보 부분을 건너뛸 수 있도록 허용합니다. 예를 들어, `T.axis.remap`을 사용하면 반복자 크기 주석을 짧게 할 수 있습니다.

TVMScript는 변환 중간에 텐서 함수를 검사하는 데도 유용한 방법입니다. 일부 상황에서는 스크립트를 출력하고 수동 편집을 한 다음 MLC 프로세스에 다시 공급하여 가능한 변환을 디버깅하고 (수동으로) 시도한 다음 MLC 프로세스에 반영하는 것이 도움이 될 수 있습니다.

#### Tensor Expression을 사용하여 TensorIR 코드 생성

많은 경우 개발 형태는 루프 수준이 아닌 더 높은 수준의 추상화입니다. 그래서 TensorIR을 얻는 또 다른 일반적인 방법은 프로그래밍 방식으로 관련 코드를 생성하는 것입니다.

Tensor expression(te)은 표현식과 유사한 API를 통해 일련의 계산을 설명하는 도메인별 언어입니다.

```{.python .input n=26}
from tvm import te
A = te.placeholder((128, 128), "float32", name="A")
B = te.placeholder((128, 128), "float32", name="B")
k = te.reduce_axis((0, 128), "k")
Y = te.compute((128, 128), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="Y")
C = te.compute((128, 128), lambda i, j: te.max(Y[i, j], 0), name="C")
```

여기서 `te.compute`는 `te.compute(output_shape, fcompute)` 시그니처를 취합니다.
fcompute 함수는 주어진 인덱스에 대해 각 요소 `Y[i, j]`의 값을 어떻게 계산하고 싶은지 설명합니다.

```python
lambda i, j: te.sum(A[i, k] * B[k, j], axis=k)
```

위의 람다 표현식은 $Y_{ij} = \sum_k A_{ik} B_{kj}$ 계산을 설명합니다. 계산을 설명한 후, 관심 있는 관련 매개변수를 전달하여 TensorIR 함수를 생성할 수 있습니다. 이 특정 경우, 두 개의 입력 매개변수(A, B)와 하나의 출력 매개변수(C)를 가진 함수를 생성하려고 합니다.

```{.python .input n=27}
te_func = te.create_prim_func([A, B, C]).with_attr({"global_symbol": "mm_relu"})
MyModuleFromTE = tvm.IRModule({"mm_relu": te_func})
IPython.display.Code(MyModuleFromTE.script(), language="python")
```

tensor expression API는 주어진 더 높은 수준의 입력에 대해 TensorIR 함수를 생성하는 유용한 도구를 제공합니다.

### 변환의 결과로서의 TensorIR 함수

실제로 변환의 결과로 TensorIR 함수를 얻기도 합니다. 이는 두 개의 원시 텐서 함수(mm과 relu)로 시작하여 프로그래밍 변환을 적용하여 단일 원시 텐서 함수 `mm_relu`로 "융합"할 때 발생합니다. 자세한 내용은 향후 장에서 다룰 것입니다.


### 토론

이 섹션에서는 지금까지 배운 내용을 검토해 보겠습니다. 우리는 일반적인 MLC 프로세스가 일련의 프로그램 변환을 따른다는 것을 배웠습니다. TensorIR 변환 프로세스를 저수준 numpy 참조 개발 프로세스와 비교하는 것은 흥미롭습니다.

![](../img/standard_process.png)

위 그림은 표준 개발 프로세스를 보여줍니다. 다른 프로그램 변형을 개발하고 그 다음 (컴파일 언어인 경우 빌드하고) 관심 있는 플랫폼에서 실행하는 프로세스를 반복해야 합니다.

MLC 프로세스(아래 그림에 표시)의 핵심 차이점은 IRModule(프로그램) 간의 프로그래밍 변환입니다. 그래서 개발을 통해(코드를 수동으로 작성하거나 코드를 생성하여) 프로그램 변형을 만들어낼 빈만 아니라 텐서 프로그램을 변환하여 변형을 얻을 수도 있습니다.

변환은 개발 비용을 단순화하고 프로세스에 더 많은 자동화를 도입하는 데 도움이 되는 매우 강력한 도구입니다. 이 섹션에서는 TensorIR을 통한 원시 텐서 함수에 대한 특정 관점을 다루었으며, 향후 더 많은 관점을 다룰 것입니다.

![](../img/mlc_process.png)


특히, 직접적인 코드 개발과 변환은 실제로 동등하게 중요합니다: 우리는 여전히 많은 도메인 전문 지식을 활용하여 프로그램의 일부를 개발하고 최적화한 다음 변환 기반 접근 방식과 결합할 수 있습니다. 향후 장에서 두 관행을 결합하는 방법에 대해 이야기할 것입니다.

### 요약

- TensorIR 추상화
  - 루프, 다차원 버퍼와 같은 일반적인 요소들을 포함
  - 루프 계산 요구사항을 캡슐화하는 새로운 구조인 **블록**을 도입
  - Python AST로 구성 가능(TVMScript를 통해)
- 변환을 사용하여 TensorIR의 다양한 변형을 생성할 수 있습니다.
- 일반적인 MLC 플로우: 개발, 변환, 빌드.
