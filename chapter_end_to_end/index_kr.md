# 엔드투엔드 모델 실행

## 서론

![](../img/tensor_func_linear_relu.png)

대부분의 MLC 프로세스는 텐서 함수 간의 변환으로 볼 수 있습니다. 우리가 다음에서 답하고자 하는 주요 질문은:

- 텐서 함수를 표현하기 위한 가능한 추상화는 무엇인가.
- 텐서 함수 간의 가능한 변환은 무엇인가.

지난 강의에서는 원시 텐서 함수에 집중했습니다. 이번 강의에서는 엔드투엔드 모델을 구축하는 방법에 대해 이야기하겠습니다.

## 준비 사항

시작하기 위해 필요한 의존성을 가져오고 헬퍼 함수를 생성하겠습니다.

```{.python .input n=1}
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T, relax as R
import numpy as np
from tvm import relax
import IPython
```

### 데이터셋 로드

구체적인 예로, fashion MNIST 데이터셋의 모델을 사용하겠습니다. 다음 코드는 `torchvision`에서 데이터를 다운로드하고 NumPy 배열로 준비합니다.

```{.python .input n=1}
import torchvision
import torch
test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

img, label = next(iter(test_loader))
img = img.reshape(1, 28, 28).numpy()
```

예측하고자 하는 이미지 인스턴스를 플롯할 수 있습니다.

```{.python .input n=2}
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(img[0])
plt.colorbar()
plt.grid(False)
plt.show()
print("Class:", class_names[label[0]])
```

### 모델 파라미터 다운로드

```{.python .input n=3}
# Hide outputs
!wget https://github.com/mlc-ai/web-data/raw/main/models/fasionmnist_mlp_params.pkl
```

## 엔드투엔드 모델 통합

이 장에서는 다음 모델을 예제로 사용하겠습니다. 이것은 relu 활성화를 가진 두 개의 선형 연산으로 구성된 2층 신경망입니다. 간단하게 유지하기 위해 최종 softmax 레이어를 제거했습니다. 출력 점수는 정규화되지 않았지만, 여전히 최대값은 가장 가능성 있는 클래스에 해당합니다.

![](../img/e2e_fashionmnist_mlp_model.png)

모델의 NumPy 구현을 검토하는 것부터 시작하겠습니다.

```{.python .input n=4}
def numpy_mlp(data, w0, b0, w1, b1):
    lv0 = data @ w0.T + b0
    lv1 = np.maximum(lv0, 0)
    lv2 = lv1 @ w1.T + b1
    return lv2
```

```{.python .input n=5}
import pickle as pkl
mlp_params = pkl.load(open("fasionmnist_mlp_params.pkl", "rb"))
res = numpy_mlp(img.reshape(1, 784),
                mlp_params["w0"],
                mlp_params["b0"],
                mlp_params["w1"],
                mlp_params["b1"])
print(res)
pred_kind = res.argmax(axis=1)
print(pred_kind)
print("NumPy-MLP Prediction:", class_names[pred_kind[0]])
```

위 예제 코드는 엔드투엔드 모델 실행을 수행하기 위한 고수준 배열 연산을 보여줍니다.

다시 MLC의 관점에서, 우리는 이러한 배열 계산의 내부 세부 사항을 살펴보고 싶습니다.

내부 세부 사항을 설명하기 위해, 다시 저수준 numpy로 예제를 작성하겠습니다:

- 가능한 루프 계산을 보여주기 위해 필요할 때 배열 함수 대신 루프를 사용합니다.
- 가능한 경우, 항상 numpy.empty를 통해 배열을 명시적으로 할당하고 전달합니다.

아래 코드 블록은 동일한 모델의 저수준 numpy 구현을 보여줍니다.


```{.python .input n=6}
def lnumpy_linear0(X: np.ndarray, W: np.ndarray, B: np.ndarray, Z: np.ndarray):
    Y = np.empty((1, 128), dtype="float32")
    for i in range(1):
        for j in range(128):
            for k in range(784):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + X[i, k] * W[j, k]

    for i in range(1):
        for j in range(128):
            Z[i, j] = Y[i, j] + B[j]


def lnumpy_relu0(X: np.ndarray, Y: np.ndarray):
     for i in range(1):
        for j in range(128):
            Y[i, j] = np.maximum(X[i, j], 0)

def lnumpy_linear1(X: np.ndarray, W: np.ndarray, B: np.ndarray, Z: np.ndarray):
    Y = np.empty((1, 10), dtype="float32")
    for i in range(1):
        for j in range(10):
            for k in range(128):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + X[i, k] * W[j, k]

    for i in range(1):
        for j in range(10):
            Z[i, j] = Y[i, j] + B[j]


def lnumpy_mlp(data, w0, b0, w1, b1):
    lv0 = np.empty((1, 128), dtype="float32")
    lnumpy_linear0(data, w0, b0, lv0)

    lv1 = np.empty((1, 128), dtype="float32")
    lnumpy_relu0(lv0, lv1)

    out = np.empty((1, 10), dtype="float32")
    lnumpy_linear1(lv1, w1, b1, out)
    return out

result =lnumpy_mlp(
    img.reshape(1, 784),
    mlp_params["w0"],
    mlp_params["b0"],
    mlp_params["w1"],
    mlp_params["b1"])

pred_kind = result.argmax(axis=1)
print("Low-level Numpy MLP Prediction:", class_names[pred_kind[0]])
```

## TVMScript로 엔드투엔드 IRModule 구성

저수준 NumPy 예제를 염두에 두고, 이제 엔드투엔드 모델 실행을 위한 MLC 추상화를 소개할 준비가 되었습니다. 아래 코드 블록은 모델의 TVMScript 구현을 보여줍니다.

```{.python .input n=7}
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def relu0(x: T.handle, y: T.handle):
        n = T.int64()
        X = T.match_buffer(x, (1, n), "float32")
        Y = T.match_buffer(y, (1, n), "float32")
        for i, j in T.grid(1, n):
            with T.block("Y"):
                vi, vj = T.axis.remap("SS", [i, j])
                Y[vi, vj] = T.max(X[vi, vj], T.float32(0))

    @T.prim_func
    def linear0(x: T.handle,
                w: T.handle,
                b: T.handle,
                z: T.handle):
        m, n, k = T.int64(), T.int64(), T.int64()
        X = T.match_buffer(x, (1, m), "float32")
        W = T.match_buffer(w, (n, m), "float32")
        B = T.match_buffer(b, (n, ), "float32")
        Z = T.match_buffer(z, (1, n), "float32")
        Y = T.alloc_buffer((1, n), "float32")
        for i, j, k in T.grid(1, n, m):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
        for i, j in T.grid(1, n):
            with T.block("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] = Y[vi, vj] + B[vj]

    @R.function
    def main(x: R.Tensor((1, "m"), "float32"),
             w0: R.Tensor(("n", "m"), "float32"),
             b0: R.Tensor(("n", ), "float32"),
             w1: R.Tensor(("k", "n"), "float32"),
             b1: R.Tensor(("k", ), "float32")):
        m, n, k = T.int64(), T.int64(), T.int64()
        with R.dataflow():
            lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, n), "float32"))
            lv1 = R.call_dps_packed("relu0", (lv0, ), R.Tensor((1, n), "float32"))
            out = R.call_dps_packed("linear0", (lv1, w1, b1), R.Tensor((1, k), "float32"))
            R.output(out)
        return out
```

위 코드에는 여러 종류의 함수가 포함되어 있습니다: 지난 강의에서 본 원시 텐서 함수(`T.prim_func`)와 새로운 `R.function`(relax 함수)입니다. Relax 함수는 고수준 신경망 실행을 나타내는 새로운 유형의 추상화입니다.

다시, TVMScript 코드와 저수준 numpy 코드를 나란히 보고 해당 요소들을 확인하는 것이 도움이 되며, 각각을 자세히 살펴보겠습니다. 원시 텐서 함수에 대해서는 이미 배웠으므로, 고수준 실행 부분에 집중하겠습니다.

![](../img/e2e_compare_to_lnumpy.png)

### 계산 그래프 뷰

![](../img/e2e_computational_graph_call_tir.png)

그래프를 사용하여 고수준 모델 실행을 시각화하는 것이 일반적으로 도움이 됩니다. 위 그림은 `main` 함수의 그래프 뷰입니다:

- 그래프의 각 박스는 계산 연산에 해당합니다.
- 화살표는 중간 텐서의 입력-출력에 해당합니다.

이전 강의에서 이런 종류의 시각화를 본 적이 있습니다. 그래프 자체는 일종의 추상화로 볼 수 있으며, 머신러닝 프레임워크에서 일반적으로 **계산 그래프**로 알려져 있습니다.


### `call_dps_packed` 구조

여러분이 알아차릴 수 있는 한 가지는 계산 그래프의 각 연산 단계가 `R.call_dps_packed` 연산을 포함한다는 것입니다. 이것은 텐서 원시 함수를 가져오는 연산입니다.

```python
lv0 = R.call_dps_packed(linear0, (x, w0, b0), (1, 128), dtype="float32")
```

`R.call_dps_packed`가 무엇을 의미하는지 설명하기 위해, 다음과 같이 연산의 동등한 저수준 numpy 구현을 검토하겠습니다:

```{.python .input n=8}
def lnumpy_call_dps_packed(prim_func, inputs, shape, dtype):
    res = np.empty(shape, dtype=dtype)
    prim_func(*inputs, res)
    return res
```

구체적으로, call_dps_packed는 원시 함수(`prim_func`)와 입력 목록을 받습니다. 그런 다음 출력 텐서 `res`를 할당하고, 입력과 출력을 `prim_func`에 전달합니다. `prim_func`를 실행한 후 결과가 `res`에 채워지고, 그런 다음 결과를 반환할 수 있습니다.

`lnumpy_call_dps_packed`는 `R.call_dps_packed`의 의미를 보여주기 위한 참조 구현일 뿐입니다. 실제로는 실행을 최적화하는 다양한 저수준 방법이 있을 수 있습니다. 예를 들어, 모든 출력 메모리를 미리 할당한 다음 실행을 수행하는 것을 선택할 수 있으며, 이는 향후 강의에서 다룰 것입니다.

자연스럽게 물어볼 수 있는 질문은 왜 `call_dps_packed` 구조가 필요한가입니다. 이는 우리의 원시 텐서 함수가 다음과 같은 호출 규칙을 취하기 때문입니다.

```python
def low_level_prim_func(in0, in1, ..., out):
    # 구현
```
이 규칙을 **destination passing**이라고 합니다. 아이디어는 입력과 출력이 명시적으로 외부에서 할당되고 저수준 원시 함수에 전달된다는 것입니다. 이 스타일은 저수준 라이브러리 설계에서 일반적으로 사용되므로, 고수준 프레임워크가 메모리 할당 결정을 처리할 수 있습니다. 모든 텐서 연산을 이 스타일로 표현할 수 있는 것은 아닙니다(구체적으로, 출력 모양이 입력에 의존하는 연산들이 있습니다). 그럼에도 불구하고, 일반적인 관행에서는 가능할 때 이 스타일로 저수준 함수를 작성하는 것이 일반적으로 도움이 됩니다.

중간 결과를 명시적으로 할당하고 각 함수를 호출하여 destination passing 규칙 함수를 함께 조립하는 것은 가능하지만, 다음 코드를 계산 그래프 형태로 변환하기는 어렵습니다.

```python
def lnumpy_mlp(data, w0, b0, w1, b1):
    lv0 = np.empty((1, 128), dtype="float32")
    lnumpy_linear0(data, w0, b0, lv0)

    lv1 = np.empty((1, 128), dtype="float32")
    lnumpy_relu0(lv0, lv1)

    out = np.empty((1, 10), dtype="float32")
    lnumpy_linear1(lv1, w1, b1, out)
    return out
```

![](../img/e2e_computational_graph_numpy.png)

한 번 시도해 볼 수는 있습니다 :) 위 그림은 단순히 함수 입력을 함수에 연결하여 `lnumpy_mlp`를 "계산 그래프 유사" 형태로 맞추려는 한 가지 "실패한 시도"입니다.

이전 계산 그래프의 몇 가지 좋은 속성을 잃어버렸음을 알 수 있습니다. 구체적으로, 계산 그래프는 일반적으로 다음과 같은 속성을 가집니다:

- 박스로 들어가는 모든 입력 간선은 연산에 대한 입력에 해당합니다.
- 모든 나가는 간선은 연산의 출력에 해당합니다.
- 각 연산은 간선의 위상 순서까지 임의로 재정렬될 수 있습니다.

물론 입력 간선과 출력 간선을 도입하여 그래프 정의를 일반화할 수는 있지만, 그러면 추상화와 관련된 가능한 변환이 복잡해질 수 있습니다.

따라서 `call_dps_packed`로 돌아가면, 여기서 핵심 통찰력은 가능한 할당이나 함수에 대한 명시적 쓰기를 숨기고 싶다는 것입니다. 보다 공식적인 용어로, 함수가 **순수(pure)** 하거나 **부작용 없는(side-effect free)** 것을 원합니다.

함수가 **순수** 하거나 **부작용이 없다**는 것은: 입력에서만 읽고 출력을 통해 결과를 반환하며, 프로그램의 다른 부분을 변경하지 않는다는 것입니다(예: 전역 카운터 증가).

**call_dps_packed**는 저수준 원시 함수 호출의 세부 사항을 숨기고 이를 계산 그래프에 노출하는 방법입니다.

저수준 numpy에서도 `call_dps_packed`를 실행으로 볼 수 있습니다. 이제 `lnumpy_call_dps_packed`를 정의했으므로, 저수준 numpy 실행 코드를 다음과 같이 다시 작성할 수 있습니다:

```{.python .input n=9}
def lnumpy_mlp_with_call_dps_packed(data, w0, b0, w1, b1):
    lv0 = lnumpy_call_dps_packed(lnumpy_linear0, (data, w0, b0), (1, 128), dtype="float32")
    lv1 = lnumpy_call_dps_packed(lnumpy_relu0, (lv0, ), (1, 128), dtype="float32")
    out = lnumpy_call_dps_packed(lnumpy_linear1, (lv1, w1, b1), (1, 10), dtype="float32")
    return out

result = lnumpy_mlp_with_call_dps_packed(
    img.reshape(1, 784),
    mlp_params["w0"],
    mlp_params["b0"],
    mlp_params["w1"],
    mlp_params["b1"])

pred_kind = np.argmax(result, axis=1)
print("Low-level Numpy with CallTIR Prediction:", class_names[pred_kind[0]])
```

실제로 최저수준 구현에는 명시적인 메모리 할당이 있으므로, `call_dps_packed`는 주로 실제 구현을 생성하기 전에 몧 가지 고수준 변환을 계속하기 위한 목적으로 사용됩니다.

### 데이터플로우 블록

relax 함수의 또 다른 중요한 요소는 `R.dataflow()` 범위 주석입니다.

```python
with R.dataflow():
    lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, n), "float32"))
    lv1 = R.call_dps_packed("relu0", (lv0, ), R.Tensor((1, n), "float32"))
    out = R.call_dps_packed("linear0", (lv1, w1, b1), R.Tensor((1, k), "float32"))
    R.output(out)
```

이것은 지난 섹션에서 논의한 **계산 그래프**로 다시 연결됩니다. 이상적으로 각 계산 그래프 연산은 부작용이 없어야 한다는 것을 기억하세요.

그런데 부작용을 포함하는 연산을 도입하고 싶다면 어떻게 할까요? 데이터플로우 블록은 프로그램의 계산 그래프 영역을 표시하는 방법입니다. 구체적으로, 데이터플로우 블록 내에서는 모든 연산이 부작용이 없어야 합니다. 데이터플로우 블록 외부에서는 연산이 부작용을 포함할 수 있습니다. 아래 프로그램은 두 개의 데이터플로우 블록을 포함하는 예제 프로그램입니다.

```python
@R.function
def main(x: R.Tensor((1, "m"), "float32"),
        w0: R.Tensor(("n", "m"), "float32"),
        b0: R.Tensor(("n", ), "float32"),
        w1: R.Tensor(("k", "n"), "float32"),
        b1: R.Tensor(("k", ), "float32")):
    m, n, k = T.int64(), T.int64(), T.int64()

    with R.dataflow():
        lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, n), "float32"))
        gv0 = R.call_dps_packed("relu0", (lv0, ), R.Tensor((1, n), "float32"))
        R.output(gv0)

    with R.dataflow():
        out = R.call_dps_packed("linear0", (gv0, w1, b1), R.Tensor((1, k), "float32"))
        R.output(out)
    return out
```

대부분의 강의는 계산 그래프(데이터플로우 블록)만 다룰 것입니다. 하지만 그 이유를 염두에 두는 것이 좋습니다.

### 섹션 체크포인트

지금까지 relax 프로그램의 한 가지 예시를 살펴보고 다음을 포함한 대부분의 요소를 다루었습니다:

- 계산 그래프 뷰
- `call_dps_packed` 구조
- 데이터플로우 블록

이러한 요소들은 엔드투엔드 모델 실행 및 컴파일을 시작하는 데 필요합니다. 나중 장에서 새로운 개념을 만나면 다룰 것입니다.

## 모델 빌드 및 실행

지난 섹션에서는 엔드투엔드 모델 실행을 표현할 수 있게 해주는 추상화에 대해 논의했습니다. 이 섹션에서는 IRModule을 빌드하고 실행하는 방법을 소개합니다. 우리가 가진 IRModule을 검토하는 것부터 시작하겠습니다.

```{.python .input n=10}
IPython.display.Code(MyModule.script(), language="python")
```

`relax.build`를 호출하여 이 함수를 빌드합니다. Relax는 아직 개발 중이므로 일부 API가 변경될 수 있습니다. 하지만 우리의 주요 목표는 엔드투엔드 모델을 위한 전체 MLC 플로우(구성, 변환, 빌드)에 익숙해지는 것입니다.

```{.python .input n=11}
ex = relax.build(MyModule, target="llvm")
type(ex)
```

빌드 함수는 실행 가능한 파일을 제공합니다. 함수를 실행할 수 있게 해주는 가상 머신 실행기를 초기화할 수 있습니다. 추가로, 엔드투엔드 실행을 어떤 장치에서 실행할지를 나타내는 두 번째 인자를 전달합니다.

```{.python .input n=12}
vm = relax.VirtualMachine(ex, tvm.cpu())
```

이제 모델을 실행할 준비가 되었습니다. 입력 데이터와 가중치를 포함하는 tvm NDArray를 구성하는 것부터 시작합니다.

```{.python .input n=13}
data_nd = tvm.nd.array(img.reshape(1, 784))
nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}
```

그런 다음 입력 인자와 가중치를 전달하여 main 함수를 실행할 수 있습니다.

```{.python .input n=14}
nd_res = vm["main"](data_nd,
                    nd_params["w0"],
                    nd_params["b0"],
                    nd_params["w1"],
                    nd_params["b1"])
print(nd_res)
```

main 함수는 예측 결과를 반환하며, `nd_res.numpy()`를 호출하여 numpy 배열로 변환하고 argmax를 취하여 클래스 레이블을 얻을 수 있습니다.

```{.python .input n=15}
pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MyModule Prediction:", class_names[pred_kind[0]])
```

## 환경의 기존 라이브러리 통합

지난 섹션에서는 원시 함수 구현과 고수준 계산 그래프 부분을 모두 포함하는 IRModule을 빌드하는 방법을 보여주었습니다. 많은 경우 기존 라이브러리 함수를 MLC 프로세스에 통합하는 데 관심이 있을 수 있습니다.

IRModule은 그 방법을 보여주는 예제를 보여줍니다.

```{.python .input n=16}
@tvm.script.ir_module
class MyModuleWithExternCall:
    @R.function
    def main(x: R.Tensor((1, "m"), "float32"),
             w0: R.Tensor(("n", "m"), "float32"),
             b0: R.Tensor(("n", ), "float32"),
             w1: R.Tensor(("k", "n"), "float32"),
             b1: R.Tensor(("k", ), "float32")):
        # block 0
        m, n, k = T.int64(), T.int64(), T.int64()
        with R.dataflow():
            lv0 = R.call_dps_packed("env.linear", (x, w0, b0), R.Tensor((1, n), "float32"))
            lv1 = R.call_dps_packed("env.relu", (lv0, ), R.Tensor((1, n), "float32"))
            out = R.call_dps_packed("env.linear", (lv1, w1, b1), R.Tensor((1, k), "float32"))
            R.output(out)
        return out
```

이제 `call_dps_packed`에 문자열을 직접 전달합니다.

```python
R.call_dps_packed("env.linear", (x, w0, b0), R.Tensor((1, n), "float32"))
```

이러한 문자열은 모델 실행 중에 존재할 것으로 예상되는 런타임 함수의 이름입니다.

### 런타임 함수 등록

외부 함수를 호출하는 코드를 실행하려면 해당 함수를 등록해야 합니다. 아래 코드 블록은 두 가지 함수의 구현을 등록합니다.

```{.python .input n=17}
@tvm.register_func("env.linear", override=True)
def torch_linear(x: tvm.nd.NDArray,
                 w: tvm.nd.NDArray,
                 b: tvm.nd.NDArray,
                 out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    w_torch = torch.from_dlpack(w)
    b_torch = torch.from_dlpack(b)
    out_torch = torch.from_dlpack(out)
    torch.mm(x_torch, w_torch.T, out=out_torch)
    torch.add(out_torch, b_torch, out=out_torch)

@tvm.register_func("env.relu", override=True)
def lnumpy_relu(x: tvm.nd.NDArray,
                out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    out_torch = torch.from_dlpack(out)
    torch.maximum(x_torch, torch.Tensor([0.0]), out=out_torch)
```

위 코드에서 `from_dlpack`을 사용하여 TVM NDArray를 torch NDArray로 변환합니다. 이는 복사가 없는(zero-copy) 변환으로, torch 배열이 TVM NDArray와 기본 메모리를 공유한다는 의미입니다. DLPack은 다른 프레임워크가 데이터 복사 없이 Tensor/NDArray를 교환할 수 있게 하는 일반적인 교환 표준입니다. `from_dlpack` API는 여러 프레임워크에서 지원되며 Python 배열 API 표준의 일부입니다. 관심이 있다면 [여기](https://dmlc.github.io/dlpack/latest/python_spec.html)에서 더 읽어보세요.

이 특정 함수에서는 단순히 PyTorch의 구현을 활용합니다. 실제 설정에서는 유사한 메커니즘을 사용하여 cuDNN이나 자체 라이브러리 구현과 같은 특정 라이브러리로 호출을 리디렉션할 수 있습니다.

이 특정 예제는 Python에서 등록을 수행합니다. 실제로는 Python 종속성이 없는 다른 언어(C++)로 함수를 등록할 수 있습니다. 향후 강의에서 더 다룰 것입니다.

### 빌드 및 실행

이제 `MyModuleWithExternCall`을 빌드하고 실행할 수 있으며, 동일한 결과를 얻는지 확인할 수 있습니다.

```{.python .input n=18}
ex = relax.build(MyModuleWithExternCall, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

nd_res = vm["main"](data_nd,
                    nd_params["w0"],
                    nd_params["b0"],
                    nd_params["w1"],
                    nd_params["b1"])

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MyModuleWithExternCall Prediction:", class_names[pred_kind[0]])
```

## TensorIR 코드와 라이브러리 혼합

지난 예제에서는 모든 원시 연산이 라이브러리 함수로 디스패치되는 IRModule을 구축했습니다. 때로는 둘 다를 혼합하는 것이 도움이 될 수 있습니다.

```{.python .input n=19}
@tvm.script.ir_module
class MyModuleMixture:
    @T.prim_func
    def linear0(x: T.handle,
                w: T.handle,
                b: T.handle,
                z: T.handle):
        m, n, k = T.int64(), T.int64(), T.int64()
        X = T.match_buffer(x, (1, m), "float32")
        W = T.match_buffer(w, (n, m), "float32")
        B = T.match_buffer(b, (n, ), "float32")
        Z = T.match_buffer(z, (1, n), "float32")
        Y = T.alloc_buffer((1, n), "float32")
        for i, j, k in T.grid(1, n, m):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
        for i, j in T.grid(1, n):
            with T.block("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] = Y[vi, vj] + B[vj]

    @R.function
    def main(x: R.Tensor((1, "m"), "float32"),
             w0: R.Tensor(("n", "m"), "float32"),
             b0: R.Tensor(("n", ), "float32"),
             w1: R.Tensor(("k", "n"), "float32"),
             b1: R.Tensor(("k", ), "float32")):
        m, n, k = T.int64(), T.int64(), T.int64()
        with R.dataflow():
            lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, n), "float32"))
            lv1 = R.call_dps_packed("env.relu", (lv0, ), R.Tensor((1, n), "float32"))
            out = R.call_dps_packed("env.linear", (lv1, w1, b1), R.Tensor((1, k), "float32"))
            R.output(out)
        return out
```

위 코드 블록은 linear0는 여전히 TensorIR로 구현되고 나머지 함수들은 라이브러리 함수로 리디렉션되는 예를 보여줍니다. 빌드하고 실행하여 결과를 검증할 수 있습니다.

```{.python .input n=20}
ex = relax.build(MyModuleMixture, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

nd_res = vm["main"](data_nd,
                    nd_params["w0"],
                    nd_params["b0"],
                    nd_params["w1"],
                    nd_params["b1"])

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MyModuleMixture Prediction:", class_names[pred_kind[0]])
```

## IRModule에 파라미터 바인드

지금까지의 모든 예제에서 명시적으로 파라미터를 전달하여 main 함수를 구성했습니다. 많은 경우 파라미터를 IRModule에 첨부된 상수로 바인드하는 것이 일반적으로 도움이 됩니다. 다음 코드는 nd_params의 키에 파라미터 이름을 매칭하여 바인딩을 생성했습니다.

```{.python .input n=21}
MyModuleWithParams = relax.transform.BindParams("main", nd_params)(MyModuleMixture)
IPython.display.Code(MyModuleWithParams.script(), language="python")
```

위 스크립트에서 `meta[relay.Constant][0]`는 상수를 저장하는 암묵적 딕셔너리에 해당합니다(스크립트의 일부로 표시되지는 않지만 여전히 IRModule의 일부입니다). 변환된 IRModule을 빌드하면 이제 입력 데이터만 전달하여 함수를 호출할 수 있습니다.

```{.python .input n=22}
ex = relax.build(MyModuleWithParams, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

nd_res = vm["main"](data_nd)

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MyModuleWithParams Prediction:", class_names[pred_kind[0]])
```

## 토론

이 장에서는 엔드투엔드 모델 실행을 설명하는 여러 방법에 대해 논의했습니다. 우리가 주목할 수 있는 한 가지는 **추상화와 구현**이라는 주제로 돌아왔다는 것입니다.

- TensorIR 함수와 라이브러리 함수 모두 동일한 destination passing 스타일을 따릅니다. 그 결과, 예제에서 하나에서 다른 하나로 호출을 단순히 교체할 수 있습니다.
- MLC 프로세스의 다른 단계에서 계산을 표현하는 다른 방법을 사용할 수 있습니다.

지금까지 엔드투엔드 IRModule을 변환하는 몇 가지 방법(예: 파라미터 바인딩)에 대해 다루었습니다. MLC의 다음 공통 주제로 돌아가겠습니다: MLC 프로세스는 실행을 가능한 다른 추상화로 표현하고 그것들 간에 변환하는 것에 관한 것입니다.

![](../img/mlc_process.png)

엔드투엔드 실행에는 많은 가능한 변환이 있습니다. 예를 들어, MyModuleMixture의 TensorIR 함수를 가져와 지난 강의에서 배운 스케줄 연산을 사용하여 `linear0` 함수를 변경할 수 있습니다. 다른 경우에는 고수준 모델 실행을 라이브러리 함수 호출과 TensorIR 함수의 혼합으로 변환하고 싶을 수 있습니다.

연습으로, IRModule에서 수행하고 싶은 변환의 종류에 대해 생각해 볼 시간을 가져보세요. 향후에도 더 많은 변환을 다룰 것입니다.

이 장에서는 IRModule을 수동으로 구성했습니다. 실제로 실제 신경망 모델은 수백 개의 레이어를 포함할 수 있으므로 수동으로 작성하는 것은 비현실적입니다. 그럼에도 불구하고 스크립트 형식은 무슨 일이 일어나는지 살펴보고 대화형 개발을 하는 데 도움이 됩니다. 향후 에피소드에서 IRModule을 프로그래밍 방식으로 구성하는 더 많은 방법에 대해서도 배울 것입니다.

## 요약
- 계산 그래프 추상화는 엔드투엔드 실행을 위해 원시 텐서 함수를 함께 연결하는 데 도움이 됩니다.
- relax 추상화의 핵심 요소는 다음을 포함합니다
  - destination passing 스타일 원시 함수를 계산 그래프에 내장하는 call_dps_packed 구조
  - 데이터플로우 블록
- 계산 그래프는 환경 라이브러리 함수와 TensorIR 함수 모두를 호출할 수 있습니다.