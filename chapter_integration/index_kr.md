# 머신러닝 프레임워크와의 통합

## 서문

이전 장들에서 우리는 머신러닝 컴파일을 위한 추상화와 텐서 함수 간의 변환에 대해 배웠습니다.

이 장에서는 기존 ML 프레임워크에서 머신러닝 모델을 MLC 플로우로 가져오는 방법에 대해 논의하겠습니다.

## 준비

먼저 필요한 의존성을 가져오겠습니다.

```{.python .input}
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T, relax as R
from tvm import relax
import numpy as np
```

```{.python .input}
import torch
import torch.nn as nn
from torch import fx
from torch.nn import functional as F
```

## Builder를 통한 IRModule 구축

이전 장들에서 우리는 TVMScript를 직접 작성하여 IRModule을 구축했습니다. 모델이 커질수록 IRModule을 구축하는 프로그래밍 방식이 필요합니다. 이 섹션에서는 해당 프로세스를 지원하는 일부 도구를 검토하겠습니다.

### TensorIR 생성을 위한 Tensor Expression

먼저 TensorIR 함수를 구축하기 위한 tensor expression 도메인 특화 언어를 검토합니다.

```{.python .input}
from tvm import te
```

TensorIR 함수에 대한 입력을 나타내는 placeholder 객체를 생성하는 것으로 시작합니다.

```{.python .input}
A = te.placeholder((128, 128), name="A", dtype="float32")
B = te.placeholder((128, 128), name="B", dtype="float32")
```

여기서 각 입력 및 중간 결과는 `te.Tensor` 객체로 표현됩니다.

```{.python .input}
type(A)
```

각 `te.Tensor`는 계산의 형상과 데이터 타입을 추적하는 shape 필드와 dtype 필드를 가지고 있습니다.

```{.python .input}
A.shape
```

tensor expression 계산의 시퀀스를 통해 계산을 설명할 수 있습니다. 여기서 `te.compute`는 시그니처 `te.compute(output_shape, fcompute)`를 사용합니다. fcompute 함수는 주어진 인덱스에 대해 각 요소 `[i, j]`의 값을 계산하는 방법을 설명합니다.

`te_matmul` 함수는 `te.Tensor` 타입의 객체를 받아 행렬 곱셈 결과를 반환합니다. A와 B의 입력 형상에 따라 계산을 어떻게 구축하는지 주목하십시오. `te_matmul`은 다양한 입력 형상을 가진 A와 B에 대해 작동합니다.

```{.python .input}
def te_matmul(A: te.Tensor, B: te.Tensor) -> te.Tensor:
    assert A.shape[1] == B.shape[0]
    n = A.shape[0]
    m = B.shape[1]
    k = te.reduce_axis((0, A.shape[1]), name="k")
    return te.compute((n, m), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="matmul")
```

A와 B로 `te_matmul`을 호출하여 matmul의 결과를 생성할 수 있습니다.

```{.python .input}
C = te_matmul(A, B)
```

TensorIR 함수를 생성하려면 `te.create_prim_func`를 호출하고 입력 및 출력 값을 전달하면 됩니다.

```{.python .input}
te.create_prim_func([A, B, C]).show()
```

유사한 방식으로 relu 계산을 위한 tensor expression을 생성할 수 있습니다. 여기서는 `te_relu` 함수가 모든 차원과 형상의 `te.Tensor`에 대해 작동하도록 작성합니다.

```{.python .input}
def te_relu(A: te.Tensor) -> te.Tensor:
    return te.compute(A.shape, lambda *i: te.max(A(*i), 0), name="relu")
```

두 가지 다른 입력 형상과 차원에 대해 `te_relu`를 시도해보겠습니다. 먼저 형상 `(10,)`을 가진 `X1`입니다.

```{.python .input}
X1 = te.placeholder((10,), name="X1", dtype="float32")
Y1 = te_relu(X1)
te.create_prim_func([X1, Y1]).show()
```

그 다음 형상 `(10, 20)`을 가진 `X2`입니다.

```{.python .input}
X2 = te.placeholder((10, 20), name="X1", dtype="float32")
Y2 = te_relu(X2)
te.create_prim_func([X2, Y2]).show()
```

`te` API가 허용하는 마지막 한 가지는 연산을 구성하고 "융합된" 연산자를 생성하는 것입니다. 예를 들어, matmul의 결과를 가져와서 다시 relu를 적용할 수 있습니다.

```{.python .input}
C = te_matmul(A, B)
D = te_relu(C)
```

관심 있는 입력 및 출력 값만 전달하고 중간 값을 건너뛰어 TensorIR 함수를 생성할 수 있습니다. 이렇게 하면 matmul의 결과가 TensorIR 함수에서 임시 공간으로 할당됩니다.

```{.python .input}
te.create_prim_func([A, B, D]).show()
```

중간 결과 C를 인수 목록에 전달할 수도 있습니다. 이 경우 TensorIR 함수는 호출자 측에서 C의 버퍼도 전달하기를 기대합니다. 일반적으로 내부에서 더 고급 융합을 가질 수 있도록 입력/출력만 전달하는 것을 권장합니다.

```{.python .input}
te.create_prim_func([A, B, C, D]).show()
```

### BlockBuilder를 사용하여 IRModule 생성

지금까지 단일 TensorIR 함수를 생성했습니다. 엔드투엔드 모델 실행을 구축하려면 계산 그래프를 통해 여러 TensorIR 함수를 연결할 수 있어야 합니다.

먼저 `relax.Function`을 점진적으로 구축하는 데 도움이 되는 block builder를 생성하겠습니다.

```{.python .input}
A = relax.Var("A", relax.TensorStructInfo((128, 128), "float32"))
B = relax.Var("B", relax.TensorStructInfo((128, 128), "float32"))
```

block builder를 생성한 다음 일련의 원시 텐서 연산을 통해 relax 함수를 구성합니다.

```{.python .input}
bb = relax.BlockBuilder()

with bb.function("main"):
    with bb.dataflow():
        C = bb.emit_te(te_matmul, A, B)
        D = bb.emit_te(te_relu, C)
        R = bb.emit_output(D)
    bb.emit_func_output(R, params=[A, B])

MyModule = bb.get()
MyModule.show()
```

### Block Builder API 상세 분석

이제 각 block builder API를 자세히 파고들어가보겠습니다. block builder 코드와 결과 모듈을 나란히 놓는 것이 도움이 됩니다.

![](../img/integration_block_builder.png)

block builder는 relax 함수의 스코프에 해당하는 스코프를 제공합니다. 예를 들어 `bb.dataflow()`는 스코프 내부의 모든 block builder 호출이 dataflow 스코프에 속하는 dataflow 블록을 생성합니다.

```python
with bb.function("main"):
    with bb.dataflow():
        # every emit call generates a variable inside a dataflow block.
```

각 중간 결과는 계산 결과를 저장하는 변수에 해당하는 `relax.Var`입니다. `DataflowVar`는 해당 변수가 dataflow 블록(계산 그래프) 내부의 중간 단계임을 나타냅니다.

```{.python .input}
type(C)
```

```{.python .input}
isinstance(C, relax.Var)
```

relax 함수의 각 줄은 `emit_te` 호출에 의해 생성됩니다. 예를 들어,

```python
lv = R.call_dps_packed(te_matmul, (A, B), (128, 128), dtype="float32")
```

는 다음에 의해 생성됩니다.

```python
C = bb.emit_te(te_matmul, A, B).
```

내부적으로 bb.emit_te는 다음을 수행합니다:

- A와 B에 대한 입력 `te.placeholder` 생성
- 그것들을 `te_matmul` 함수에 통과시킴
- `te.create_prim_func`를 호출하여 TensorIR 함수 생성
- `call_dps_packed`를 통해 함수 호출 생성

결과가 두 개의 중간 값을 가진 계산 그래프이며, 하나의 노드는 te_matmul 연산에 해당하고 다른 하나는 `te_relu`에 해당하는 것을 알 수 있습니다.

`bb.emit_output`을 통해 각 dataflow 블록의 출력 변수를 생성할 수 있습니다.

```python
with bb.dataflow():
    ...
    R = bb.emit_output(D)
```

위 코드는 D가 dataflow 블록 외부에서 참조할 수 있는 변수임을 표시합니다.

마지막으로 함수 출력은 `bb.emit_func_output`으로 표시됩니다. 각 함수 스코프에서 `emit_func_output`을 한 번만 호출할 수 있습니다.

특히, 출력 방출 단계에서 함수의 파라미터 목록을 지정할 수 있습니다. 이렇게 하면 파라미터 목록을 즉석에서 수집하는 경우에 도움이 됩니다.

```python
with bb.function("main"):
    ...
    # 끝에 파라미터 지정
    bb.emit_func_output(R, params=[A, B])
```
또는 함수 스코프의 시작 부분에서 파라미터 목록을 지정할 수 있습니다.

```python
# 시작 부분에 파라미터 지정
with bb.function("main", params=[A, B]):
    ...
    bb.emit_func_output(R)
```

## PyTorch에서 모델 가져오기

이제 IRModule을 프로그래밍 방식으로 구성하는 도구를 배웠습니다. 이를 사용하여 PyTorch의 모델을 IRModule 형식으로 가져오겠습니다.

대부분의 머신러닝 프레임워크는 계산 그래프 추상화를 제공하며, 각 노드는 연산에 해당하고 엣지는 그들 간의 종속성에 해당합니다. PyTorch 모델을 가져와서 PyTorch의 네이티브 형식으로 계산 그래프를 얻은 다음 이를 IRModule로 변환하겠습니다.

PyTorch에서 모델을 정의하는 것으로 시작하겠습니다. 예제를 일관되게 유지하기 위해 matmul relu 예제를 사용하겠습니다.

```{.python .input}
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(128, 128))

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.relu(x)
        return x
```

### TorchFX GraphModule 생성

TorchFX를 사용하여 PyTorch 모듈에서 그래프를 추적합니다.

```{.python .input}
model = MyModel()
fx_module = fx.symbolic_trace(model)
type(fx_module)
```

`fx_module`은 표 데이터로 출력할 수 있는 간단한 계산 그래프 뷰를 포함합니다. 우리의 목표는 이 그래프를 IRModule로 변환하는 것입니다.

```{.python .input}
fx_module.graph.print_tabular()
```

### Map 함수 생성

전체 고수준 변환 로직을 정의하겠습니다. 주요 흐름은 다음과 같습니다:

- IRModule에서 변환된 노드를 나타내는 해당 `relax.Var`에 `fx.Node`를 매핑하는 `node_map` 생성
- 위상 순서로 fx 그래프의 노드를 반복
- 매핑된 입력을 기반으로 노드의 매핑된 출력 계산

```{.python .input}
def map_param(param: nn.Parameter):
    return relax.const(
        param.data.cpu().numpy(), relax.TensorStructInfo(param.data.shape, "float32")
    )

def fetch_attr(fx_mod, target: str):
    """Helper function to fetch an attr"""
    target_atoms = target.split('.')
    attr_itr = fx_mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr

def from_fx(fx_mod, input_shapes, call_function_map, call_module_map):
    input_index = 0
    node_map = {}
    named_modules = dict(fx_mod.named_modules())

    bb = relax.BlockBuilder()

    fn_inputs = []
    fn_output = None
    with bb.function("main"):
        with bb.dataflow():
            for node in fx_mod.graph.nodes:
                if node.op == "placeholder":
                    # create input placeholder
                    shape = input_shapes[input_index]
                    input_index += 1
                    input_var = relax.Var(
                        node.target, relax.TensorStructInfo(shape, "float32")
                    )
                    fn_inputs.append(input_var)
                    node_map[node] = input_var
                elif node.op == "get_attr":
                    node_map[node] = map_param(fetch_attr(fx_mod, node.target))
                elif node.op == "call_function":
                    node_map[node] = call_function_map[node.target](bb, node_map, node)
                elif node.op == "call_module":
                    named_module = named_modules[node.target]
                    node_map[node] = call_module_map[type(named_module)](bb, node_map, node, named_module)
                elif node.op == "output":
                    output = node_map[node.args[0]]
                    assert fn_output is None
                    fn_output = bb.emit_output(output)
        # output and finalize the function
        bb.emit_func_output(output, fn_inputs)
    return bb.get()
```

`from_fx` 함수에서 함수 맵을 정의하지 않았습니다. 각 torch 함수의 변환 규칙을 맵을 통해 제공하겠습니다. 구체적으로, 다음 코드 블록은 `emit_te` API를 통해 이를 수행하는 방법을 보여줍니다.

```{.python .input}
def map_matmul(bb, node_map, node: fx.Node):
    A = node_map[node.args[0]]
    B = node_map[node.args[1]]
    return bb.emit_te(te_matmul, A, B)

def map_relu(bb, node_map, node: fx.Node):
    A = node_map[node.args[0]]
    return bb.emit_te(te_relu, A)

MyModule = from_fx(
    fx_module,
    input_shapes = [(1, 128)],
    call_function_map = {
      torch.matmul: map_matmul,
      torch.relu: map_relu,
    },
    call_module_map={},
)

MyModule.show()
```

## FashionMNIST 예제로 돌아가기

```{.python .input}
import torch
import torchvision

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

```{.python .input}
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(img[0])
plt.colorbar()
plt.grid(False)
plt.show()

print("Class:", class_names[label[0]])
```

```{.python .input}
# Hide outputs
!wget -nc https://github.com/mlc-ai/web-data/raw/main/models/fasionmnist_mlp_params.pkl
```

![](../img/e2e_fashionmnist_mlp_model.png)

위는 관심 모델이며, 다음과 같이 PyTorch 모델을 구축할 수 있습니다.

```{.python .input}
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear0 = nn.Linear(784, 128, bias=True)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = self.linear0(x)
        x = self.relu(x)
        x = self.linear1(x)
        return x
```

```{.python .input}
import pickle as pkl
mlp_model = MLP()

mlp_params = pkl.load(open("fasionmnist_mlp_params.pkl", "rb"))
mlp_model.linear0.weight.data = torch.from_numpy(mlp_params["w0"])
mlp_model.linear0.bias.data = torch.from_numpy(mlp_params["b0"])
mlp_model.linear1.weight.data = torch.from_numpy(mlp_params["w1"])
mlp_model.linear1.bias.data = torch.from_numpy(mlp_params["b1"])
```

```{.python .input}
torch_res = mlp_model(torch.from_numpy(img.reshape(1, 784)))

pred_kind = np.argmax(torch_res.detach().numpy(), axis=1)
print("Torch Prediction:", class_names[pred_kind[0]])
```

해당 `nn.Module`에 대한 매핑 함수를 정의하여 fx로부터 변환을 시도해보겠습니다. 여기서는 우리 자체 tensor expression을 정의하는 대신 TVM `topi`에서 사전 정의된 TE 라이브러리를 재사용합니다.

- `topi.nn.dense(x, w)`는 전치 행렬 곱셈 `x @ w.T`를 수행합니다.
- `topi.add`는 브로드캠스트 덧셈을 수행합니다.

```{.python .input}
from tvm import topi

def map_nn_linear(bb, node_map, node, nn_mod):
    x = node_map[node.args[0]]
    w = map_param(nn_mod.weight)
    if nn_mod.bias is not None:
        b = map_param(nn_mod.bias)
    y = bb.emit_te(topi.nn.dense, x, w)
    return bb.emit_te(topi.add, y, b)

def map_nn_relu(bb, node_map, node, nn_mod):
    return map_relu(bb, node_map, node)


MLPModule = from_fx(
    fx.symbolic_trace(mlp_model),
    input_shapes = [(1, 784)],
    call_function_map={
    },
    call_module_map={
        torch.nn.Linear: map_nn_linear,
        torch.nn.ReLU: map_nn_relu,
    },
)

MLPModule.show()
```

```{.python .input}
ex = relax.build(MLPModule, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())
data_nd = tvm.nd.array(img.reshape(1, 784))

nd_res = vm["main"](data_nd)

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MLPModule Prediction:", class_names[pred_kind[0]])
```

## 참고: 고수준 연산자로 변환

대부분의 머신러닝 프레임워크에서는 먼저 고수준 내장 원시 연산자로 변환하는 것이 도움이 될 수 있습니다. 다음 코드 블록은 이를 수행하는 예제를 제공합니다.

```{.python .input}
def map_nn_relu_op(bb, node_map, node, nn_mod):
    A = node_map[node.args[0]]
    return bb.emit(relax.op.nn.relu(A))

def map_nn_linear_op(bb, node_map, node, nn_mod):
    x = node_map[node.args[0]]
    w = map_param(nn_mod.weight)
    b = map_param(nn_mod.bias)
    return bb.emit(relax.op.linear(x, w, b))

MLPModuleHighLevel = from_fx(
    fx.symbolic_trace(mlp_model),
    input_shapes = [(1, 784)],
    call_function_map={
    },
    call_module_map={
        torch.nn.Linear: map_nn_linear_op,
        torch.nn.ReLU: map_nn_relu_op,
    },
)

MLPModuleHighLevel.show()
```

이러한 내장 연산자 호출을 사용하여 모델을 IRModule로 가져온 후,
이러한 내장 연산자는 TensorIR 함수보다 **더 고수준의 추상화**입니다. 이러한 원시 연산자를 라이브러리 또는 TensorIR 함수로 추가 변환할 다양한 기회가 있을 수 있습니다.

대부분의 경우, 고수준 내장 함수가 사용 가능할 때 이로 변환하는 것이 도움이 될 수 있습니다. 그러나 해당하는 고수준 내장 함수를 찾을 수 없거나 TensorIR 함수를 직접 지정하려는 경우가 많습니다. 이러한 경우 변환 로직이나 변환을 사용자 정의하여 `call_dps_packed`를 생성하거나 라이브러리 함수를 호출할 수 있습니다. 일반적으로 고수준 op, TensorIR, 라이브러리 추상화를 결합하여 최상의 결과를 얻을 수 있습니다. 후속 강의에서 이러한 트레이드오프를 논의하겠습니다.

## 논의

이 장에서는 MLC 플로우의 **개발** 부분에 초점을 맞췄습니다. 머신러닝 프레임워크에서 IRModule로 모델을 가져오는 다양한 방법을 학습했습니다. 또한 고수준 원시 연산자에 대해서도 간략하게 다루었습니다.

모델을 IRModule로 가져오면 원시 함수와 계산 그래프 함수에 대한 더 많은 종류의 변환을 도입할 수 있습니다. 좋은 MLC 프로세스는 이러한 변환들을 함께 구성하여 최종 배포 형태를 형성합니다.

![](../img/mlc_process.png)

## 요약

- Tensor expression API를 사용하면 원시 TensorIR 함수를 생성할 수 있습니다.
- BlockBuilder API는 `emit_te` 및 기타 함수를 통해 IRModule을 생성합니다.
- 모델을 IRModule로 변환하여 기존 머신러닝 프레임워크와 통합합니다.
