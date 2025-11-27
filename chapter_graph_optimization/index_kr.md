# 계산 그래프 최적화

## 서문

대부분의 MLC 프로세스는 텐서 함수 간의 변환으로 볼 수 있습니다. 이전 장들에서는 각 원시 텐서 함수를 개별적으로 변환하는 방법을 학습했습니다. 이 장에서는 계산 그래프 간의 고수준 변환에 대해 논의하겠습니다.

![](../img/mlc-elem-transform.png)

## 준비

먼저 필요한 의존성을 가져오겠습니다.

```{.python .input}
# This is needed for deferring annotation parsing in TVMScript
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T, relax as R
from tvm import relax, topi
import numpy as np
```

## 패턴 매칭 및 재작성

다음 예제로 시작하겠습니다.

```{.python .input}
@tvm.script.ir_module
class MyModule:
    @R.function
    def main(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
        with R.dataflow():
            lv0 = relax.op.multiply(x, y)
            gv0 = relax.op.add(lv0, y)
            R.output(gv0)
        return gv0
```

`MyModule`은 두 개의 고수준 연산자 relax.op.multiply와 relax.op.add를 가진 relax 함수를 포함합니다. 우리의 목표는 이 두 연산자를 찾아서 `relax.op.ewise_fma` 연산자 호출로 교체하는 것입니다.

정확히 어떻게 하는지 살펴보기 전에 먼저 MyModule을 구성하는 데이터 구조를 살펴보겠습니다. 각 IRModule은 함수 모음을 포함하며, 함수 본문은 추상 구문 트리(AST)라고 하는 데이터 구조 세트로 구성됩니다.

```{.python .input}
relax_func = MyModule["main"]
```

각 함수는 `relax.expr.Function` 노드로 표현됩니다.

```{.python .input}
type(relax_func)
```

함수는 매개변수 목록을 포함합니다.

```{.python .input}
relax_func.params
```

함수는 반환 값과 함수 내 바인딩 블록 세트를 나타내는 body 필드를 포함합니다.

```{.python .input}
func_body = relax_func.body
type(func_body)
```

함수 본문 SeqExpr은 (바인딩) 블록의 시퀀스를 포함합니다.

```{.python .input}
func_body.blocks
```

```{.python .input}
dataflow_block = func_body.blocks[0]
```

우리의 특정 경우에는 두 개의 바인딩을 포함하는 단일 데이터 플로우 블록이 있습니다. 각 바인딩은 다음 두 줄 중 하나에 해당합니다.

```python
lv0 = relax.op.multiply(x, y)
gv0 = relax.op.add(lv0, y)
```

```{.python .input}
dataflow_block.bindings
```

```{.python .input}
binding = dataflow_block.bindings[0]
```

각 바인딩은 바인딩의 왼쪽(`lv0`, `gv0`)에 해당하는 var 필드를 가지고 있습니다.

```{.python .input}
binding.var
```

그리고 value 필드는 바인딩의 오른쪽에 해당합니다. 각 value 필드는 원시 함수로의 호출을 나타내는 `relax.Call` 노드에 해당합니다.

```{.python .input}
binding.value
```

![](../img/relax_func_data_structure.png)

위 그림은 이 특정 함수에 관련된 데이터 구조를 요약합니다.

프로그램을 재작성하는 한 가지 접근 방식은 MyModule의 AST를 재귀적으로 순회하고 변환된 AST를 생성하는 것입니다. 사용 가능한 Python API를 사용하여 확실히 그렇게 할 수 있습니다. 그러나 프로세스를 단순화하기 위해 추가 도구 지원을 사용할 수 있습니다. 다음 코드 블록은 각 AST 노드를 방문하고 변환된 버전으로 재작성할 수 있게 해주는 **방문자 패턴**이라는 디자인 패턴을 따릅니다.

```{.python .input}
@relax.expr_functor.mutator
class EwiseFMARewriter(relax.PyExprMutator):
    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)
        add_op = tvm.ir.Op.get("relax.add")
        multiply_op = tvm.ir.Op.get("relax.multiply")
        ewise_fma_op = tvm.ir.Op.get("relax.ewise_fma")

        if call.op != add_op:
            return call

        value = self.lookup_binding(call.args[0])
        if not isinstance(value, relax.Call) or value.op != multiply_op:
            return call

        fma_call = relax.Call(
            ewise_fma_op, [value.args[0], value.args[1], call.args[1]], None, None
        )
        return fma_call


updated_fn = EwiseFMARewriter().visit_expr(MyModule["main"])
updated_fn.show()
```

코드를 실행할 수 있습니다. 결과는 gv0를 융합된 연산자로 재작성하지만 lv0는 코드에 남겨둡니다. `remove_all_unused`를 사용하여 코드 블록을 더 단순화할 수 있습니다.

```{.python .input}
relax.analysis.remove_all_unused(updated_fn).show()
```

## 선형 및 ReLU 융합

이제 그래프 재작성의 기본적인 맛을 보았습니다. 엔드투엔드 모델에서 시도해보겠습니다.

```{.python .input}
# Hide outputs
!wget https://github.com/mlc-ai/web-data/raw/main/models/fasionmnist_mlp_params.pkl
```

```{.python .input}
import pickle as pkl
mlp_params = pkl.load(open("fasionmnist_mlp_params.pkl", "rb"))
```

다음 코드는 이전 장에서 사용한 FashionMNIST MLP 모델을 재구성합니다. 설명을 단순화하기 위해 `relax.op.add` 및 `relax.op.matmul`과 같은 고수준 연산자를 사용하여 모델을 직접 구성합니다.

```{.python .input}
def create_model():
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((1, 784), "float32"))
    w0 = relax.const(mlp_params["w0"], "float32")
    b0 = relax.const(mlp_params["b0"], "float32")
    w1 = relax.const(mlp_params["w1"], "float32")
    b1 = relax.const(mlp_params["b1"], "float32")
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.matmul(x, relax.op.permute_dims(w0)))
            lv1 = bb.emit(relax.op.add(lv0, b0))
            lv2 = bb.emit(relax.op.nn.relu(lv1))
            lv3 = bb.emit(relax.op.matmul(lv2, relax.op.permute_dims(w1)))
            lv4 = bb.emit(relax.op.add(lv3, b1))
            gv = bb.emit_output(lv4)
        bb.emit_func_output(gv)

    return bb.get()

MLPModel = create_model()
MLPModel.show()
```

우리는 dense와 add 연산을 단일 그룹으로 "융합"하는 것을 목표로 합니다. 다음 코드는 다음 단계를 통해 이를 달성합니다:

- `matmul`과 `add` 패턴을 식별합니다.
- matmul과 add 연산자를 호출하는 또 다른 융합된 하위 함수를 생성합니다.
- `matmul`과 `add`를 융합된 하위 함수로 교체합니다.

```{.python .input}
@relax.expr_functor.mutator
class MatmulAddFusor(relax.PyExprMutator):
    def __init__(self, mod: IRModule) -> None:
        super().__init__()
        self.mod_ = mod
        # cache pre-defined ops
        self.add_op = tvm.ir.Op.get("relax.add")
        self.matmul_op = tvm.ir.Op.get("relax.matmul")
        self.counter = 0

    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            # avoid already fused primitive functions
            if func.attrs is not None and "Primitive" in func.attrs.keys() and func.attrs["Primitive"] != 0:
                continue
            updated_func = self.visit_expr(func)
            updated_func = relax.analysis.remove_all_unused(updated_func)
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()

    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)

        def match_call(node, op):
            if not isinstance(node, relax.Call):
                return False
            return node.op == op

        # pattern match matmul => add
        if not match_call(call, self.add_op):
            return call

        value = self.lookup_binding(call.args[0])
        if value is None:
            return call

        if not match_call(value, self.matmul_op):
            return call

        x = value.args[0]
        w = value.args[1]
        b = call.args[1]

        # construct a new fused primitive function
        param_x = relax.Var("x" ,relax.TensorStructInfo(x.struct_info.shape, x.struct_info.dtype))
        param_w = relax.Var("w" ,relax.TensorStructInfo(w.struct_info.shape, w.struct_info.dtype))
        param_b = relax.Var("b" ,relax.TensorStructInfo(b.struct_info.shape, b.struct_info.dtype))

        bb = relax.BlockBuilder()

        fn_name = "fused_matmul_add%d" % (self.counter)
        self.counter += 1
        with bb.function(fn_name, [param_x, param_w, param_b]):
            with bb.dataflow():
                lv0 = bb.emit(relax.op.matmul(param_x, param_w))
                gv = bb.emit_output(relax.op.add(lv0, param_b))
            bb.emit_func_output(gv)

        # Add Primitive attribute to the fused funtions
        fused_fn = bb.get()[fn_name].with_attr("Primitive", 1)
        global_var = self.builder_.add_func(fused_fn, fn_name)

        # construct call into the fused function
        return relax.Call(global_var, [x, w, b], None, None)

@tvm.ir.transform.module_pass(opt_level=2, name="MatmulAddFuse")
class FuseDenseAddPass:
    """The wrapper for the LowerTensorIR pass."""
    def transform_module(self, mod, ctx):
        return MatmulAddFusor(mod).transform()


MLPFused = FuseDenseAddPass()(MLPModel)
MLPFused.show()
```

### 하위 함수를 생성하는 이유

위 예제에서 우리는 `fuse_matmul_add` 접두사를 가진 두 개의 하위 함수를 생성했습니다. 이러한 하위 함수 본문에는 융합된 연산자가 수행하는 연산에 대한 정보가 포함되어 있습니다. 이 재작성의 대안은 단순히 융합된 연산자(`ewise_fma`처럼)에 대한 별도의 원시 연산을 생성하는 것입니다. 그러나 더 많은 연산자를 융합하려고 할 때 가능한 조합의 양이 기하급수적으로 증가할 수 있습니다. 융합된 연산을 함께 그룹화하는 하위 함수는 각 융합 패턴에 대해 전용 고수준 연산자를 도입하지 않고도 후속 코드 낮추기를 위해 동일한 양의 정보를 제공합니다.

## TensorIR 호출로 매핑

융합된 IRModule은 고수준 연산으로의 호출만 포함합니다. 추가 저수준 최적화 및 코드 생성을 위해 이러한 고수준 원시 연산자를 해당 TensorIR 함수(또는 환경 라이브러리 함수)로 변환해야 합니다.

다음 코드는 고수준 연산을 해당 TensorIR 함수로 다시 매핑합니다. 여기서는 각 Mutator의 내부 블록 빌더를 활용하고 `call_te`를 사용하여 변환된 값을 반환합니다.

```{.python .input}
@relax.expr_functor.mutator
class LowerToTensorIR(relax.PyExprMutator):
    def __init__(self, mod: IRModule, op_map) -> None:
        super().__init__()
        self.mod_ = mod
        self.op_map = {
            tvm.ir.Op.get(k): v for k, v in op_map.items()
        }


    def visit_call_(self, call):
        call = self.visit_expr_post_order(call)

        if call.op in self.op_map:
            return self.op_map[call.op](self.builder_, call)
        return call

    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions.items():
            if not isinstance(func, relax.Function):
                continue
            updated_func = self.visit_expr(func)
            self.builder_.update_func(global_var, updated_func)

        return self.builder_.get()


def map_matmul(bb, call):
    x, w = call.args
    return bb.call_te(topi.nn.matmul, x, w)

def map_add(bb, call):
    a, b = call.args
    return bb.call_te(topi.add, a, b)

def map_relu(bb, call):
    return bb.call_te(topi.nn.relu, call.args[0])

def map_transpose(bb, call):
    return bb.call_te(topi.transpose, call.args[0], )

op_map = {
  "relax.matmul": map_matmul,
  "relax.add": map_add,
  "relax.nn.relu": map_relu,
  "relax.permute_dims": map_transpose
}

@tvm.ir.transform.module_pass(opt_level=0, name="LowerToTensorIR")
class LowerToTensorIRPass:
    """The wrapper for the LowerTensorIR pass."""
    def transform_module(self, mod, ctx):
        return LowerToTensorIR(mod, op_map).transform()


MLPModelTIR = LowerToTensorIRPass()(MLPFused)
MLPModelTIR.show()
```

위 코드에서 주목할 점은 `fused_matmul_add0`과 `fused_matmul_add1`이 여전히 해당 TensorIR matmul과 add 함수를 호출하는 고수준 relax 함수라는 것입니다. 이들을 단일 TensorIR 함수로 변환할 수 있으며, 이는 후속 최적화 및 코드 생성 단계에 사용될 수 있습니다.

```{.python .input}
MLPModelFinal = relax.transform.FuseTIR()(MLPModelTIR)
MLPModelFinal.show()
```

## 빌드 및 실행

최종 모듈을 빌드하고 예제 이미지에서 시도해볼 수 있습니다.

```{.python .input}
# Hide outputs
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

```{.python .output}
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(img[0])
plt.colorbar()
plt.grid(False)
plt.show()

print("Class:", class_names[label[0]])
```

```{.python .output}
ex = relax.build(MLPModelFinal, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())
data_nd = tvm.nd.array(img.reshape(1, 784))

nd_res = vm["main"](data_nd)

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MLPModule Prediction:", class_names[pred_kind[0]])
```

## 논의

이 섹션은 계산 그래프 간의 **변환**이라는 우리의 공통 주제로 돌아갑니다. 최소한이긴 하지만 이 변환 시퀀스는 MLC 프로세스에서 일반적으로 수행하는 두 가지 중요한 최적화인 융합과 루프 수준 코드 낮추기를 다룹니다.

실제 MLC 프로세스는 더 강력하고 견고한 변환을 포함할 수 있습니다. 예를 들어, 우리의 융합 패스는 dense 연산자가 두 개의 후속 add 연산에서 참조되는 중복된 dense 계산을 생성할 수 있습니다. 견고한 융합 패스는 이를 감지하고 그러한 경우를 건너뛰도록 선택할 것입니다. 또한 각 조합에 대한 규칙을 작성할 필요가 없습니다. 대신 TVM의 내부 퓨저는 TensorIR 함수 루프 패턴을 분석하고 융합 결정에 사용합니다.

특히 이러한 각 변환은 서로 조합 가능합니다. 예를 들어, 탐색하고자 하는 추가적인 새로운 융합 패턴을 지원하기 위해 사용자 정의 퓨저 버전을 사용하도록 선택한 다음 나머지 단계를 처리하기 위해 기존 퓨저에 공급할 수 있습니다.

![](../img/mlc_process.png)

## 요약

- 계산 그래프 데이터 구조를 재작성하여 텐서 프로그램을 최적화할 수 있습니다.
- 호출 노드를 재작성하기 위한 방문자 패턴.
- 융합 및 루프 수준 프로그램 낮추기와 같은 계산 그래프 변환을 수행할 수 있습니다.
