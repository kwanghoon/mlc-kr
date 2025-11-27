---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3.8.5 64-bit
    language: python
    name: python3
---

# MLC 과제 1: 엔드투엔드 모델 실행


## 섹션 1: 모델 준비

MLC를 사용하여 엔드투엔드 모델을 구축하고 조작하는 과정에 익숙해지기 위해, 간단한 이미지 분류 모델부터 시작하겠습니다.

먼저 다음 명령어를 사용하여 필요한 패키지를 설치합니다.

```python
!python3 -m pip install mlc-ai-nightly -f https://mlc.ai/wheels
!python3 -m pip install torch torchvision torchaudio torchsummary --extra-index-url https://download.pytorch.org/whl/cpu
```

```python
import numpy as np
import pickle as pkl
import torch
import torch.nn.functional as F
import torchvision
import tvm
import tvm.testing

from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from tvm import topi, relax, te
from tvm.script import tir as T

```

아래는 PyTorch로 정의된 모델입니다. 이미지 배치를 입력으로 받아들여 합성곱 레이어, 활성화 레이어, 풀링 레이어, 전체 연결 레이어를 순서대로 통과시킵니다.

```python
batch_size = 4
input_shape = (batch_size, 1, 28, 28)  # NCHW layout


def pytorch_model():
    list = []
    list.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), bias=True))
    list.append(nn.ReLU())
    list.append(nn.MaxPool2d(kernel_size=(2, 2)))
    list.append(nn.Flatten())
    list.append(nn.Linear(in_features=5408, out_features=100, bias=True))
    list.append(nn.ReLU())
    list.append(nn.Linear(in_features=100, out_features=10, bias=True))
    list.append(nn.Softmax(dim=1))

    model = nn.Sequential(*list).cpu()
    name_map = {
        "0.weight": "conv2d_weight",
        "0.bias": "conv2d_bias",
        "4.weight": "linear0_weight",
        "4.bias": "linear0_bias",
        "6.weight": "linear1_weight",
        "6.bias": "linear1_bias",
    }
    for name, param in model.named_parameters():
        param.data = torch.from_numpy(weight_map[name_map[name]]).cpu()
    return model

```

Fashion MNIST 데이터셋에서 이 모델에 대한 사전 학습된 가중치 맵을 제공합니다.

```python
# Hide outputs
!wget -nc https://github.com/mlc-ai/web-data/raw/main/models/fasionmnist_mlp_assignment_params.pkl
```

정확도가 약 84%임을 확인할 수 있습니다.

```python
# Load the weight map from file.
# The prediction accuracy of the weight map on test data is around 83.3%.
weight_map = pkl.load(open("fasionmnist_mlp_assignment_params.pkl", "rb"))
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        print_img = True
        for data, label in test_loader:
            data, label = data.cpu(), label.cpu()
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, label, reduction="sum").item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            if print_img:
                imshow(data[0])
                print("predict: {}, label: {}".format(class_names[pred[0][0]], class_names[label[0]]))
                print_img = False
            correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


test_data = torchvision.datasets.FashionMNIST(
    "./data",
    download=True,
    train=False,
    transform=transforms.Compose([transforms.ToTensor()])
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False)
test(pytorch_model(), test_loader)

```

## 섹션 2. PyTorch로부터 모델 가져오기

엔드투엔드 모델의 MLC 추상화를 보려면 PyTorch로부터 모델을 가져와 TVMScript 구현으로 변환해야 합니다. 그러나 이를 수동으로 수행하기는 어렵습니다. 연습 문제 1에서 경험했듯이, 각 모델 레이어에 대한 원시 텐서 함수를 작성하려면 막대한 엔지니어링 노력이 필요합니다. 게다가 수동 작성 프로세스는 오류가 발생하기 쉽습니다 - 수십 줄의 코드를 작성하는데 구현에 작은 버그가 있다고 상상해 보면, 버그를 찾는 것이 짜증날 수 있습니다.

다행히도 TVM에는 훨씬 더 간단한 방법이 있습니다. TVM은 빈 IRModule에서 시작하여 단계별로 엔드투엔드 모델을 구축할 수 있는 `relax.BlockBuilder` 유틸리티를 제공합니다. (강의 4에서 계산 그래프 수준의 MLC 추상화인 Relax의 데이터플로우 블록 설계를 소개했습니다. 여기서 "`BlockBuilder`"의 "블록"은 Relax 함수의 데이터플로우 블록을 의미합니다.)

구체적으로, `BlockBuilder`에는 강의 3에서 소개된 Tensor Expression 연산자 설명을 해당 연산자의 TensorIR 함수에 대한 `call_tir` 연산으로 변환하는 데 도움을 주는 `emit_te` API가 있습니다(`call_tir`도 강의 4에서 소개되었습니다). TensorIR 함수를 수동으로 작성하는 것과 비교하여, Tensor Expression 설명을 작성하는 것은 몇 줄의 코드만으로 수행할 수 있어 노력의 양을 줄이고 실수할 가능성이 적습니다.

`emit_te`의 시그니처는 `emit_te(func, *input)`이며, 여기서 `func`는 Tensor Expression 연산자 설명을 반환하는 함수이고, `*input`은 `func`에 대한 입력입니다.

소개 예제로 시작하겠습니다. 아래 코드 블록에서 `relu`는 ReLU 연산자의 Tensor Expression 설명을 반환하는 함수입니다. 단일 ReLU 연산자를 실행하는 Relax 함수를 구성하기 위해, `emit_te_example` 함수에서 먼저 BlockBuilder 인스턴스 `bb`를 정의합니다. 또한 ReLU 연산의 입력 텐서(그리고 Relax 함수의 입력)로 사용될 2차원 128x128 텐서 변수 `x`를 정의합니다.

그 후, `with bb.function(name, [*input])` API를 사용하여 `x`를 입력으로 하는 Relax 함수 `main`을 구성합니다. 그런 다음 데이터플로우 블록을 구성합니다. 데이터플로우 블록 내부에서 먼저 `emit_te`를 통해 ReLU 연산자의 TensorIR 구현에 대한 `call_tir`을 가집니다. 아래의 `emit_te`는 IRModule에 "`relu`"라는 TensorIR 함수를 생성하고, 데이터플로우 블록에 `call_tir(relu, (x,), (128, 128), dtype="float32")` 연산을 추가합니다. 그리고 `call_tir` 뒤에 함수 반환이 이어집니다.

이 구성 후, BlockBuilder `bb`는 구성된 IRModule을 포함하며, 이는 `bb.get()`으로 얻을 수 있습니다.


```python
def relu(A):
    B = te.compute(shape=(128, 128), fcompute=lambda i, j: te.max(A[i, j], 0), name="B")
    return B


def emit_te_example():
    bb = relax.BlockBuilder()
    x = relax.Var("x", (128, 128), relax.DynTensorType(2, "float32"))
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit_te(relu, x)
            gv = bb.emit_output(lv0)
        bb.emit_func_output(gv)
    return bb.get()

```

`emit_te_example` 함수는 구성된 IRModule을 출력으로 반환합니다. BlockBuilder가 무엇을 구성하는지 확인하기 위해 IRModule을 출력합니다.

```python
import IPython

mod = emit_te_example()
IPython.display.Code(mod.script(), language="python")

```

보시다시피, BlockBuilder가 생성한 IRModule에는 ReLU의 TensorIR 구현과 `call_tir`을 통해 ReLU 구현을 호출하는 Relax 함수가 포함되어 있습니다.

이제 위에서 정의한 PyTorch 모델과 동등한 IRModule을 생성하기 위해 BlockBuilder와 `emit_te`를 사용할 차례입니다. 모든 연산자에 대한 Tensor Expression 설명을 직접 작성할 수 있습니다. 또는 TVM은 다양한 연산자에 대한 Tensor Expression 설명을 래핑하는 TOPI("TVM Operator Inventory"의 약자) 라이브러리를 제공합니다. [문서](https://tvm.apache.org/docs/reference/api/python/topi.html)를 읽고 사용 방법을 찾아볼 것을 권장합니다. IRModule의 정확성을 쉽게 확인할 수 있도록 테스트 함수가 제공되었습니다.

모델의 각 Conv2d 레이어 또는 linear 레이어에는 bias add가 포함되어 있으므로, 구성하는 IRModule에 이가 반영되어야 합니다.

```python
def create_model_via_emit_te():
    bb = relax.BlockBuilder()
    x = relax.Var("x", input_shape, relax.DynTensorType(batch_size, "float32"))

    conv2d_weight = relax.const(weight_map["conv2d_weight"], "float32")
    conv2d_bias = relax.const(weight_map["conv2d_bias"].reshape(1, 32, 1, 1), "float32")
    linear0_weight = relax.const(weight_map["linear0_weight"], "float32")
    linear0_bias = relax.const(weight_map["linear0_bias"].reshape(1, 100), "float32")
    linear1_weight = relax.const(weight_map["linear1_weight"], "float32")
    linear1_bias = relax.const(weight_map["linear1_bias"].reshape(1, 10), "float32")

    with bb.function("main", [x]):
        with bb.dataflow():
            # TODO
            ...
        bb.emit_func_output(gv)

    return bb.get()


def build_mod(mod):
    exec = relax.vm.build(mod, "llvm")
    dev = tvm.cpu()
    vm = relax.VirtualMachine(exec, dev)
    return vm


def check_equivalence(mod, torch_model, test_loader):
    torch_model.eval()
    with torch.no_grad():
        rt_mod = build_mod(mod)
        for data, label in test_loader:
            data, label = data.cpu(), label.cpu()
            output_from_pytorch = torch_model(data).numpy()
            output_from_relax = rt_mod["main"](tvm.nd.array(data, tvm.cpu())).numpy()
            tvm.testing.assert_allclose(output_from_pytorch, output_from_relax, rtol=1e-4)


test_data = torchvision.datasets.FashionMNIST(
    "./data",
    download=True,
    train=False,
    transform=transforms.Compose([transforms.ToTensor()])
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

mod = create_model_via_emit_te()
torch_model = pytorch_model()

check_equivalence(mod, torch_model, test_loader)
IPython.display.Code(mod.script(), language="python")

```

<!-- #region -->
## 섹션 3. 벤더 라이브러리 사용

강의 4에서 논의한 바와 같이, IRModule에 torch 함수를 통합할 수 있습니다. 단계에는 외부 런타임 함수를 등록하고 IRModule 내부에서 `call_tir`을 사용하여 호출하는 것이 포함됩니다.

다음은 torch matmul과 torch add를 사용하여 선형 레이어를 구현하는 예제입니다. 이 예제는 강의 4 노트에서도 찾을 수 있습니다.

```python
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


@tvm.script.ir_module
class MyModuleWithExternCall:
    @R.function
    def main(x: Tensor((1, 784), "float32"),
             w0: Tensor((128, 784), "float32"),
             b0: Tensor((128,), "float32")):
        # block 0
        with R.dataflow():
            lv0 = R.call_tir("env.linear", (x, w0, b0), (1, 128), dtype="float32")
            ...
        return ...
```
<!-- #endregion -->

섹션 2에서 생성한 IRModule에 나타나는 합성곱 레이어에 대한 외부 함수를 등록하십시오. 함수의 구현으로 NumPy 또는 PyTorch를 사용해야 합니다.

구성 중인 Relax 함수의 끝에 `call_tir` 연산을 직접 추가하려면 `BlockBuilder.emit`을 사용할 수 있습니다.

```python

def create_model_with_torch_func():
    bb = relax.BlockBuilder()

    x = relax.Var("x", input_shape, relax.DynTensorType(4, "float32"))

    conv2d_weight = relax.const(weight_map["conv2d_weight"], "float32")
    conv2d_bias = relax.const(weight_map["conv2d_bias"].reshape(1, 32, 1, 1), "float32")
    linear0_weight = relax.const(weight_map["linear0_weight"], "float32")
    linear0_bias = relax.const(weight_map["linear0_bias"].reshape(1, 100), "float32")
    linear1_weight = relax.const(weight_map["linear1_weight"], "float32")
    linear1_bias = relax.const(weight_map["linear1_bias"].reshape(1, 10), "float32")

    with bb.function("main", [x]):
        with bb.dataflow():
            # TODO
            ...
        bb.emit_func_output(gv)

    return bb.get()


test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
mod = create_model_with_torch_func()
check_equivalence(mod, torch_model, test_loader)

```

## 섹션 4. 엔드투엔드 모델의 변환

연습 문제 1에서 단일 TensorIR 함수를 변환하는 방법을 배웠습니다. 엔드투엔드 모델에서 이를 수행하는 것도 비슷합니다.

배치 matmul 프로그램과 비교하여, 더 도전적인 conv2d에 집중하겠습니다.

먼저 몸 가지 새로운 기본 요소를 소개하겠습니다:
 - `compute_inline`: 메모리 사용량과 메모리 접근을 줄이기 위해 블록을 다른 블록에 인라인합니다.
 - `fuse`: `split`의 반대입니다. 여러 축을 융합합니다. 여기서 `fuse`는 `parallel` / `vectorize` / `unroll`과 함께 사용되어 병렬성을 더욱 높입니다.

```python
@T.prim_func
def before_inline(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.alloc_buffer((128, 128))
    C = T.match_buffer(c, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0
    for i, j in T.grid(128, 128):
        with T.block("C"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = B[vi, vj] + 1.0


sch = tvm.tir.Schedule(before_inline)
sch.compute_inline(sch.get_block("B"))
IPython.display.Code(sch.mod["main"].script(), language="python")

```

```python
@T.prim_func
def before_fuse(a: T.handle, b: T.handle) -> None:
    A = T.match_buffer(a, (128, 128))
    B = T.match_buffer(b, (128, 128))
    for i, j in T.grid(128, 128):
        with T.block("B"):
            vi, vj = T.axis.remap("SS", [i, j])
            B[vi, vj] = A[vi, vj] * 2.0


sch = tvm.tir.Schedule(before_fuse)
i, j = sch.get_loops(sch.get_block("B"))
sch.fuse(i, j)
IPython.display.Code(sch.mod["main"].script(), language="python")

```

<!-- #region -->
이제 먼저 IRModule에 대한 스케줄을 생성한 다음 내부의 conv2d TensorIR 함수를 변환합니다. 연습 문제 1과 유사하게 타겟 함수를 제공합니다. 하지만 타겟 함수는 여러 가지 이유로 "표준 변환 답안"으로 사용되지 않는다는 점에 유의하십시오:
 - 모든 하드웨어에서 최상의 성능을 발휘하지 않을 수 있습니다.
 - 섹션 2에서 사용한 Tensor Expression 설명에 따라 원래 conv2d TensorIR 구현이 달라질 수 있습니다:
   - Tensor Expression에서 conv2d 계산과 bias 계산을 함께 설명한 경우, 타겟 TensorIR 함수의 끝에 bias를 계산하는 블록이 있어야 합니다.
   - conv2d와 bias 계산을 별도로 설명했거나 TOPI에서 제공하는 conv2d를 사용한 경우, 타겟 함수의 끝에 bias 블록이 없어야 합니다. 타겟의 원본 함수는 TOPI conv2d를 사용하여 생성되었습니다.


```python
@T.prim_func
def target_func(rxplaceholder: T.Buffer[(4, 1, 28, 28), "float32"], rxplaceholder_1: T.Buffer[(32, 1, 3, 3), "float32"], conv2d_nchw: T.Buffer[(4, 32, 26, 26), "float32"]) -> None:
    T.func_attr({"global_symbol": "conv2d", "tir.noalias": True})
    # body
    # with T.block("root")
    for i0_0_i1_0_i2_0_i3_0_fused in T.parallel(2704):
        for i0_1_i1_1_fused_init in T.unroll(8):
            for i2_1_i3_1_fused_init in T.vectorized(4):
                with T.block("conv2d_nchw_init"):
                    nn = T.axis.spatial(
                        4, i0_0_i1_0_i2_0_i3_0_fused // 1352 * 2 + i0_1_i1_1_fused_init // 4)
                    ff = T.axis.spatial(
                        32, i0_0_i1_0_i2_0_i3_0_fused % 1352 // 169 * 4 + i0_1_i1_1_fused_init % 4)
                    yy = T.axis.spatial(
                        26, i0_0_i1_0_i2_0_i3_0_fused % 169 // 13 * 2 + i2_1_i3_1_fused_init // 2)
                    xx = T.axis.spatial(
                        26, i0_0_i1_0_i2_0_i3_0_fused % 13 * 2 + i2_1_i3_1_fused_init % 2)
                    T.reads()
                    T.writes(conv2d_nchw[nn, ff, yy, xx])
                    conv2d_nchw[nn, ff, yy, xx] = T.float32(0)
        for i4, i5, i6 in T.grid(1, 3, 3):
            for i0_1_i1_1_fused in T.unroll(8):
                for i2_1_i3_1_fused in T.vectorized(4):
                    with T.block("conv2d_nchw_update"):
                        nn = T.axis.spatial(
                            4, i0_0_i1_0_i2_0_i3_0_fused // 1352 * 2 + i0_1_i1_1_fused // 4)
                        ff = T.axis.spatial(
                            32, i0_0_i1_0_i2_0_i3_0_fused % 1352 // 169 * 4 + i0_1_i1_1_fused % 4)
                        yy = T.axis.spatial(
                            26, i0_0_i1_0_i2_0_i3_0_fused % 169 // 13 * 2 + i2_1_i3_1_fused // 2)
                        xx = T.axis.spatial(
                            26, i0_0_i1_0_i2_0_i3_0_fused % 13 * 2 + i2_1_i3_1_fused % 2)
                        rc, ry, rx = T.axis.remap("RRR", [i4, i5, i6])
                        T.reads(conv2d_nchw[nn, ff, yy, xx], rxplaceholder[nn,
                                rc, yy + ry, xx + rx], rxplaceholder_1[ff, rc, ry, rx])
                        T.writes(conv2d_nchw[nn, ff, yy, xx])
                        conv2d_nchw[nn, ff, yy, xx] = conv2d_nchw[nn, ff, yy, xx] + \
                            rxplaceholder[nn, rc, yy + ry, xx +
                                          rx] * rxplaceholder_1[ff, rc, ry, rx]
```
<!-- #endregion -->

연습 문제 1과 달리, 이번에는 TensorIR 함수 대신 IRModule에 대해 스케줄이 생성됩니다. 따라서 `sch.get_block`을 사용할 때 아래와 같이 구체적인 함수 이름을 제공해야 합니다.

```python
mod = create_model_via_emit_te()
sch = tvm.tir.Schedule(mod)
# TODO
# Step 1. Get blocks
# block = sch.get_block(name="your_block_name", func_name="your_function_name")

# Step 2. Inline the padding block (if exists)

# Step 3. Get loops

# Step 4. Organize the loops

# Step 5. decompose reduction

# Step 6. fuse + vectorize / fuse + parallel / fuse + unroll

IPython.display.Code(sch.mod.script(), language="python")

```

다시, 변환된 IRModule의 정확성을 테스트할 수 있습니다.

```python
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
check_equivalence(sch.mod, torch_model, test_loader)

```
