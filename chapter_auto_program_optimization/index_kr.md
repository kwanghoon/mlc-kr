# 자동 프로그램 최적화

## 서문

이전 장들에서 우리는 원시 텐서 함수를 구축하고 이를 연결하여 엔드투엔드 모델 실행을 형성하는 방법을 배웠습니다. 지금까지 사용한 세 가지 주요 추상화 유형이 있습니다.

- 고수준 실행을 구동하는 계산 그래프 뷰
- 원시 텐서 함수에 대한 추상화
- 환경 함수 등록을 통한 라이브러리 함수 호출

이러한 모든 요소는 IRModule에 캡슐화됩니다. 대부분의 MLC 프로세스는 텐서 함수 간의 변환으로 볼 수 있습니다.

동일한 프로그램을 변환하는 방법은 여러 가지가 있습니다. 이 장에서는 일부 프로세스를 자동화하는 방법에 대해 논의하겠습니다.

## 준비

먼저 필요한 의존성을 가져오고 헬퍼 함수를 만들겠습니다.

```{.python .input n=0}
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T, relax as R
import numpy as np
from tvm import relax
```

```{.python .input n=1}
import IPython

def code2html(code):
    """Helper function to use pygments to turn the code string into highlighted html."""
    import pygments
    from pygments.lexers import Python3Lexer
    from pygments.formatters import HtmlFormatter
    formatter = HtmlFormatter()
    html = pygments.highlight(code, Python3Lexer(), formatter)
    return "<style>%s</style>%s\n" % (formatter.get_style_defs(".highlight"), html)
```

## 복습: 원시 텐서 함수 변환

이전 장에서 수행한 내용, 즉 단일 원시 텐서 함수를 변환하는 것을 복습하며 시작하겠습니다.

```{.python .input n=2}
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(
        A: T.Buffer((128, 128), "float32"),
        B: T.Buffer((128, 128), "float32"),
        C: T.Buffer((128, 128), "float32"),
    ):
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(128, 128, 128):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

먼저 평가를 위한 입력과 출력 세트를 정의하겠습니다.

```{.python .input n=3}
dtype = "float32"
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)
c_mm = a_np @ b_np
```

다음과 같이 `MyModule`을 빌드하고 실행할 수 있습니다.

```{.python .input n=4}
a_nd = tvm.nd.array(a_np)
b_nd = tvm.nd.array(b_np)
c_nd = tvm.nd.empty((128, 128), dtype="float32")

lib = tvm.build(MyModule, target="llvm")
f_timer_before = lib.time_evaluator("main", tvm.cpu())
print("Time cost of MyModule: %.3f ms" % (f_timer_before(a_nd, b_nd, c_nd).mean * 1000))
```

다음으로 루프 접근 패턴을 재구성하여 `MyModule`을 약간 변환합니다.

```{.python .input n=5}
def schedule_mm(sch: tvm.tir.Schedule, jfactor=4):
    block_C = sch.get_block("C", "main")
    i, j, k = sch.get_loops(block=block_C)
    j_0, j_1 = sch.split(loop=j, factors=[None, jfactor])
    sch.reorder(i, j_0, k, j_1)
    sch.decompose_reduction(block_C, k)
    return sch
```

```{.python .input n=6}
sch = tvm.tir.Schedule(MyModule)
sch = schedule_mm(sch)
IPython.display.HTML(code2html(sch.mod.script()))
```

그런 다음 재구성된 프로그램을 빌드하고 실행할 수 있습니다.

```{.python .input n=7}
lib = tvm.build(sch.mod, target="llvm")
f_timer_after = lib.time_evaluator("main", tvm.cpu())
print("Time cost of MyModule=>schedule_mm: %.3f ms" % (f_timer_after(a_nd, b_nd, c_nd).mean * 1000))
```

### 변환 추적

`sch.mod` 필드 외에도 `tir.Schedule`이 제공하는 또 다른 것은 변환된 모듈에 도달하는 데 관련된 단계를 보여주는 데 사용할 수 있는 trace 필드입니다. 다음 코드를 사용하여 출력할 수 있습니다.

```{.python .input n=8}
print(sch.trace)
```

```{.python .input n=9}
def schedule_mm(sch: tvm.tir.Schedule, jfactor=4):
    block_C = sch.get_block("C", "main")
    i, j, k = sch.get_loops(block=block_C)
    j_0, j_1 = sch.split(loop=j, factors=[None, jfactor])
    sch.reorder(i, j_0, k, j_1)
    sch.decompose_reduction(block_C, k)
    return sch
```

위의 trace는 `schedule_mm`에서 지정한 변환과 일치합니다. 주목할 점은 trace(와 원래 프로그램)가 최종 출력 프로그램을 완전히 재파생할 수 있는 방법을 제공한다는 것입니다. 이 점을 기억하십시오. 이 장 전체에서 trace를 변환을 검사하는 또 다른 방법으로 사용할 것입니다.

## 확률적 스케줄 변환

지금까지 우리는 원래 TensorIR 프로그램에 수행하려는 변환에 대한 모든 세부 사항을 지정했습니다. 이러한 선택의 대부분은 캐시 및 하드웨어 유닛과 같은 기본 환경에 대한 우리의 이해를 기반으로 합니다.

그러나 실제로는 모든 세부 사항을 정확하게 결정할 수 없을 수 있습니다. 그렇게 하는 대신, **프로그램을 변환하는 가능한 방법을 지정하면서 일부 세부 사항은 남겨두고 싶습니다**.

이 목표를 달성하는 자연스러운 방법은 변환에 확률적(무작위성) 요소를 추가하는 것입니다. 다음 코드가 이를 수행합니다.

```{.python .input n=10}
def stochastic_schedule_mm(sch: tvm.tir.Schedule):
    block_C = sch.get_block("C", "main")
    i, j, k = sch.get_loops(block=block_C)
    j_factors = sch.sample_perfect_tile(loop=j, n=2)
    j_0, j_1 = sch.split(loop=j, factors=j_factors)
    sch.reorder(i, j_0, k, j_1)
    sch.decompose_reduction(block_C, k)
    return sch
```

![](../img/auto_prog_optim_stoch_sch_transformation.png)

`stochastic_schedule_mm`과 `schedule_mm`을 나란히 비교해보겠습니다. 유일한 차이점은 `j_factors`를 지정하는 방법입니다. `schedule_mm`의 경우 `j_factors`는 우리가 지정한 매개변수로 전달됩니다. `stochastic_schedule_mm`의 경우에는 `sch.sample_perfect_tile`에서 나옵니다.

이름에서 알 수 있듯이 `sch.sample_perfect_tile`은 `j_factors`를 채우기 위해 난수를 추출하려고 시도합니다. 루프를 완벽하게 분할하는 인수를 샘플링합니다. 예를 들어, 원래 루프 크기가 `128`일 때, 루프를 분할하는 가능한 방법에는 `[8, 16]`, `[32, 4]`, `[2, 64]`가 포함됩니다(`8 * 16 = 32 * 4 = 2 * 64 = 128` 참고).

먼저 다음 코드 블록을 실행하여 `stochastic_schedule_mm`의 효과를 확인해보겠습니다. 다음 코드 블록을 여러 번 실행하고 결과의 차이를 관찰하십시오. 코드 블록을 실행할 때마다 `j_1`의 루프 경계가 변경되는 것을 발견할 수 있습니다.

```{.python .input n=11}
sch = tvm.tir.Schedule(MyModule)
sch = stochastic_schedule_mm(sch)

IPython.display.HTML(code2html(sch.mod.script()))
```

여기서 일어나는 일은 `stochastic_schedule_mm`을 실행할 때마다 다른 `j_factors`를 무작위로 추출한다는 것입니다. 최신 trace를 출력하여 샘플링에서 내린 결정을 확인할 수 있습니다.

```{.python .input n=12}
print(sch.trace)
```

trace를 볼 때 `sample_perfect_tile`의 `decision=[...]` 부분에 특히 주의를 기울이십시오. 이것은 `stochastic_schedule_mm`에 대한 마지막 호출에서 `sampling_perfect_tile`이 선택한 값에 해당합니다.

`stochastic_schedule_mm`의 다른 샘플을 보는 대체 방법으로, 다음 블록을 여러 번 실행하고 trace를 확인할 수 있습니다.

```{.python .input n=13}
sch = tvm.tir.Schedule(MyModule)
sch = stochastic_schedule_mm(sch)
print(sch.trace)
```

### 확률적 변환의 상세 분석

이제 확률적 스케줄 변환에서 일어난 일을 더 깊이 파고들어가보겠습니다. 이것은 두 가지 추가 요소가 있는 원래 결정론적 변환의 간단한 일반화임을 알 수 있습니다:

- `sample_perfect_tile` 및 예제에서 다루지 않은 다른 샘플링 연산에서 나오는 난수 변수
- 난수 변수에 따라 동작을 취하는 스케줄 연산

확률적 변환을 단계별로 실행해보겠습니다.

```{.python .input n=14}
sch = tvm.tir.Schedule(MyModule)
block_C = sch.get_block("C", "main")
i, j, k = sch.get_loops(block=block_C)
j_factors = sch.sample_perfect_tile(loop=j, n=2)
```

```{.python .input n=15}
type(j_factors[0])
```

`j_factors`의 요소들은 실제 정수가 아닙니다. 대신 샘플링되는 난수 변수를 참조하는 **심볼릭 변수**입니다. 이러한 변수를 변환 API에 전달하여 인수 값과 같은 선택을 지정할 수 있습니다. 

```{.python .input n=16}
print(sch.trace)
```

스케줄 trace는 이러한 심볼릭 변수의 선택을 `decisions` 필드에 추적합니다. 따라서 후속 단계에서 이러한 선택을 참조하여 루프를 분할하는 방법을 결정할 수 있습니다.

```{.python .input n=17}
IPython.display.HTML(code2html(sch.mod.script()))
```

현재 시점의 코드를 보면, 난수 변수를 샘플링했을 뿐 이를 기반으로 한 변환 동작을 아직 취하지 않았기 때문에 모듈이 동일하게 유지되는 것을 알 수 있습니다.

이제 몇 가지 동작을 취해보겠습니다:

```{.python .input n=18}
j_0, j_1 = sch.split(loop=j, factors=j_factors)
sch.reorder(i, j_0, k, j_1)
```

이러한 동작은 다음 trace에 기록됩니다.

```{.python .input n=19}
print(sch.trace)
```

코드를 다시 보면, 변환된 모듈은 이제 동작이 취해진 후의 업데이트된 버전에 해당합니다.

```{.python .input n=20}
IPython.display.HTML(code2html(sch.mod.script()))
```

최종 상태에 도달하기 위해 추가 변환을 수행할 수 있습니다.

```{.python .input n=21}
sch.reorder(i, j_0, k, j_1)
sch.decompose_reduction(block_C, k)
```

```{.python .input n=22}
IPython.display.HTML(code2html(sch.mod.script()))
```

## 확률적 변환에 대한 검색

깨달을 수 있는 한 가지는 `stochastic_schedule_mm`이 각 샘플링 단계에서 내린 특정 결정에 따라 **가능한 프로그램의 검색 공간**을 생성한다는 것입니다.

![](../img/auto_prog_optim_transformation_search.png)

초기 직관으로 돌아가면, 하나의 프로그램 대신 **가능한 프로그램의 집합**을 지정할 수 있기를 원합니다. `stochastic_schedule_mm`이 정확히 그렇게 했습니다. 물론 다음으로 물어볼 자연스러운 질문은 무엇이 최선의 선택인가 하는 것입니다.

이를 위해서는 검색 알고리즘이 필요합니다. 여기서 무엇을 할 수 있는지 보여주기 위해, 먼저 가장 간단한 검색 알고리즘인 무작위 검색을 다음 코드 블록에서 시도해보겠습니다. 이것은 `stochastic_schedule_mm`을 반복적으로 실행하고, 변환된 모듈을 얻은 다음, 벤치마크를 실행하고, 히스토리에서 최선의 것을 기록합니다.

```{.python .input n=23}
def random_search(mod: tvm.IRModule, num_trials=5):
    best_result = None
    best_sch = None

    for i in range(num_trials):
        sch = stochastic_schedule_mm(tvm.tir.Schedule(mod))
        lib = tvm.build(sch.mod, target="llvm")
        f_timer_after = lib.time_evaluator("main", tvm.cpu())
        result = f_timer_after(a_nd, b_nd, c_nd).mean

        print("=====Attempt %d, time-cost: %.3f ms====" % (i, result * 1000))
        print(sch.trace)

        # book keep the best result so far
        if best_result is None or result < best_result:
            best_result = result
            best_sch = sch      
    
    return best_sch

sch = random_search(MyModule)
```

코드를 실행하면 몇 가지 선택을 거친 다음 5번의 시도 중 최선의 실행을 반환하는 것을 알 수 있습니다.

```{.python .input n=24}
print(sch.trace)
```

실제로는 더 스마트한 알고리즘을 사용합니다. 다른 장치에 대한 최적화에 관심이 있는 경우 원격 장치에서의 벤치마킹과 같은 추가 유틸리티도 제공해야 합니다. TVM의 meta schedule API는 이러한 추가 기능을 제공합니다.

`meta_schedule`은 가능한 변환의 공간에 대한 검색을 지원하는 네임스페이스입니다. meta-schedule이 백그라운드에서 수행하는 많은 추가 작업들이 있습니다:

- 여러 프로세스에 걸친 병렬 벤치마킹
- 매번 벤치마킹을 피하기 위해 비용 모델 사용
- 매번 무작위로 샘플링하는 대신 trace에 대한 진화적 검색

이러한 마법에도 불구하고 핵심 아이디어는 동일하게 유지됩니다: **확률적 변환을 사용하여 좋은 프로그램의 검색 공간을 지정하고, `tune_tir` API가 검색 공간 내에서 최적화된 솔루션을 찾는 데 도움을 줍니다**.

```{.python .input n=25}
from tvm import meta_schedule as ms

database = ms.tune_tir(
    mod=MyModule,
    target="llvm --num-cores=1",
    max_trials_global=64,
    num_trials_per_iter=64,
    space=ms.space_generator.ScheduleFn(stochastic_schedule_mm),
    work_dir="./tune_tmp",
)

sch = ms.tir_integration.compile_tir(database, MyModule, "llvm --num-cores=1")
```

`tune_tir` 함수는 튜닝 프로세스 동안 찾은 최적화된 스케줄을 반환합니다.

```{.python .input n=26}
sch.trace.show()
```

```{.python .input n=27}
IPython.display.HTML(code2html(sch.mod.script()))
```

```{.python .input n=28}
lib = tvm.build(sch.mod, target="llvm")
f_timer_after = lib.time_evaluator("main", tvm.cpu())
print("Time cost of MyModule after tuning: %.3f ms" % (f_timer_after(a_nd, b_nd, c_nd).mean * 1000))
```

### 기본 자동 스케줄링 활용

이전 섹션에서는 우리가 만든 확률적 변환으로 워크로드를 튜닝하는 방법을 보여주었습니다. Metaschedule은 광범위한 TensorIR 계산에 작동하는 자체 내장된 일반적인 확률적 변환 세트를 제공합니다. 이 접근 방식은 검색 공간이 시스템에 의해 생성되기 때문에 자동 스케줄링이라고도 합니다. `space=ms.space_generator.ScheduleFn(stochastic_schedule_mm)` 줄을 제거하면 실행할 수 있습니다.

내부적으로 meta-scheduler는 각 블록의 데이터 접근 및 루프 패턴을 분석하고 프로그램에 확률적 변환을 제안합니다. 이 장에서는 이러한 일반적인 변환에 대해 자세히 다루지 않지만, 이들도 코드 분석과 결합된 확률적 변환일 뿐임을 주목하기 바랍니다. 이전 섹션에서 배운 동일한 메커니즘을 사용하여 자동 스케줄링을 향상시킬 수 있습니다. 향후 장에서 이 주제를 다룰 예정입니다.

```{.python .input n=29}
database = ms.tune_tir(
    mod=MyModule,
    target="llvm --num-cores=1",
    max_trials_global=64,
    num_trials_per_iter=64,
    work_dir="./tune_tmp",
)
sch = ms.tir_integration.compile_tir(database, MyModule, "llvm --num-cores=1")
```

```{.python .input n=30}
lib = tvm.build(sch.mod, target="llvm")
f_timer_after = lib.time_evaluator("main", tvm.cpu())
print("Time cost of MyModule after tuning: %.3f ms" % (f_timer_after(a_nd, b_nd, c_nd).mean * 1000))
```

결과는 원래 코드보다 훨씬 빠릅니다. trace와 최종 코드를 살펴볼 수 있습니다. 이 장의 목적상 모든 변환을 이해할 필요는 없습니다. 높은 수준에서 trace는 다음을 포함합니다:

- 더 많은 레벨의 루프 타일링 변환
- 중간 계산의 벡터화
- 루프의 병렬화 및 언롤링

```{.python .input n=31}
sch.trace.show()
```

```{.python .input n=32}
IPython.display.HTML(code2html(sch.mod.script()))
```

### 섹션 체크포인트

지금까지 배운 내용에 대한 체크포인트를 가져보겠습니다.

- 확률적 스케줄은 "가능한 변환이 무엇인지"를 표현할 수 있게 해줍니다.
- Metaschedule의 `tune_tir` API는 공간 내에서 좋은 솔루션을 찾는 데 도움을 줍니다.
- Metaschedule은 광범위한 검색 공간을 커버하는 기본 내장 확률적 변환 세트를 제공합니다.

## 엔드투엔드 모델 실행으로 다시 가져가기

지금까지 단일 텐서 원시 함수의 프로그램 최적화를 자동화하는 방법을 배웠습니다. 어떻게 이를 다시 가져와서 엔드투엔드 모델 실행을 개선할 수 있을까요?

MLC 관점에서 자동화된 검색은 모듈식 단계이며, 원래 원시 함수 구현을 튜닝 결과에서 제공하는 새로운 것으로 교체하기만 하면 됩니다.

지난 장의 2계층 MLP 예제를 재사용하겠습니다.

```{.python .input n=33}
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

```{.python .input n=34}
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(img[0])
plt.colorbar()
plt.grid(False)
plt.show()

print("Class:", class_names[label[0]])
```

예제에서 사용할 사전 패킹된 모델 파라미터도 다운로드합니다.

```{.python .input n=35}
# Hide outputs
!wget -nc https://github.com/mlc-ai/web-data/raw/main/models/fasionmnist_mlp_params.pkl
```

![](../img/e2e_fashionmnist_mlp_model.png)

참고로, 위 그림은 관심 모델을 보여줍니다.

```{.python .input n=36}
import pickle as pkl
mlp_params = pkl.load(open("fasionmnist_mlp_params.pkl", "rb"))

data_nd = tvm.nd.array(img.reshape(1, 784))
nd_params = {k: tvm.nd.array(v) for k, v in mlp_params.items()}
```

대부분의 구성 요소가 환경 함수를 호출하고 하나의 TensorIR 함수 `linear0`을 포함하는 혼합 모듈을 사용하겠습니다.

```{.python .input n=37}
@tvm.script.ir_module
class MyModuleMixture: 
    @T.prim_func
    def linear0(X: T.Buffer((1, 784), "float32"), 
                W: T.Buffer((128, 784), "float32"), 
                B: T.Buffer((128,), "float32"), 
                Z: T.Buffer((1, 128), "float32")):
        T.func_attr({"global_symbol": "linear0", "tir.noalias": True})
        Y = T.alloc_buffer((1, 128), "float32")
        for i, j, k in T.grid(1, 128, 784):
            with T.block("Y"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + X[vi, vk] * W[vj, vk]
    
        for i, j in T.grid(1, 128):
            with T.block("Z"):
                vi, vj = T.axis.remap("SS", [i, j])
                Z[vi, vj] =  Y[vi, vj] + B[vj]

    @R.function
    def main(x: R.Tensor((1, 784), "float32"), 
             w0: R.Tensor((128, 784), "float32"), 
             b0: R.Tensor((128,), "float32"), 
             w1: R.Tensor((10, 128), "float32"), 
             b1: R.Tensor((10,), "float32")):
        with R.dataflow():
            lv0 = R.call_dps_packed("linear0", (x, w0, b0), R.Tensor((1, 128), dtype="float32"))
            lv1 = R.call_dps_packed("env.relu", (lv0,), R.Tensor((1, 128), dtype="float32"))
            out = R.call_dps_packed("env.linear", (lv1, w1, b1), R.Tensor((1, 10), dtype="float32"))
            R.output(out)
        return out
```

```{.python .input n=38}
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

파라미터를 바인딩하고 올바른 예측을 제공하는지 확인할 수 있습니다.

```{.python .input n=39}
MyModuleWithParams = relax.transform.BindParams("main", nd_params)(MyModuleMixture)
```

```{.python .input n=40}
ex = relax.build(MyModuleWithParams, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

nd_res = vm["main"](data_nd)

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MyModuleWithParams Prediction:", class_names[pred_kind[0]])
```

다음 코드는 변환 전 모듈의 실행 시간 비용을 평가합니다. 이것은 작은 모델이므로 실행 간 숫자가 약간 변동할 수 있으므로 전체적인 크기만 확인하면 됩니다.

```{.python .input n=41}
ftimer = vm.module.time_evaluator("main", tvm.cpu(), number=100)

print("MyModuleWithParams time-cost: %g ms" % (ftimer(data_nd).mean * 1000))
```

이제 `linear0`을 튜닝할 준비가 되었습니다. 전체 프로세스는 다음 다이어그램에 요약되어 있습니다.

![](../img/auto_prog_optim_optim_flow.png)

현재 tune API는 하나의 `main` 함수를 가진 IRModule만 받으므로, 먼저 `linear0`을 다른 모듈의 main 함수로 추출하여 tune에 전달합니다.

```{.python .input n=42}
mod_linear = tvm.IRModule.from_expr(MyModuleMixture["linear0"].with_attr("global_symbol", "main"))
IPython.display.HTML(code2html(mod_linear.script()))
```

```{.python .input n=43}
database = ms.tune_tir(
    mod=mod_linear,
    target="llvm --num-cores=1",
    max_trials_global=64,
    num_trials_per_iter=64,
    work_dir="./tune_tmp",
)
sch = ms.tir_integration.compile_tir(database, mod_linear, "llvm --num-cores=1")
```

이제 튜닝 후 원래 `linear0`을 새 함수로 교체해야 합니다. 먼저 IRModule 내부 함수에 대한 `포인터` 참조인 `global_var`를 가져온 다음 `update_func`를 호출하여 함수를 새 것으로 교체하면 됩니다.

```{.python .input n=44}
MyModuleWithParams2 = relax.transform.BindParams("main", nd_params)(MyModuleMixture)
new_func = sch.mod["main"].with_attr("global_symbol", "linear0")
gv = MyModuleWithParams2.get_global_var("linear0")
MyModuleWithParams2.update_func(gv, new_func)
IPython.display.HTML(code2html(MyModuleWithParams2.script()))
```

위 코드에서 `linear0`이 교체된 것을 확인할 수 있습니다.

```{.python .input n=45}
ex = relax.build(MyModuleWithParams2, target="llvm")
vm = relax.VirtualMachine(ex, tvm.cpu())

nd_res = vm["main"](data_nd)

pred_kind = np.argmax(nd_res.numpy(), axis=1)
print("MyModuleWithParams2 Prediction:", class_names[pred_kind[0]])
```

코드를 다시 실행하면 주로 새로운 `linear0` 함수 덕분에 측정 가능한 시간 감소를 얻는 것을 확인할 수 있습니다.

```{.python .input n=46}
ftimer = vm.module.time_evaluator("main", tvm.cpu(), number=50)

print("MyModuleWithParams2 time-cost: %g ms" % (ftimer(data_nd).mean * 1000))
```

## 논의

이전 두 장은 **추상화**에 초점을 맞췄던 반면, 이 장은 **변환**에 초점을 맞추기 시작한다는 것을 눈치처을 수 있습니다. 확률적 변환은 모든 선택을 확정하지 않고 최적화할 수 있는 것을 지정합니다. meta-schedule API는 가능한 변환의 공간을 검색하고 최상의 것을 선택하는 데 도움을 줍니다.

중요하게도, 검색 결과를 엔드투엔드 플로우에 다시 넣는 것은 원래 함수의 구현을 튜닝 프로세스에서 알려주는 새로운 것으로 교체하는 문제일 뿐입니다.

따라서 우리는 다시 아래 그림의 일반적인 MLC 프로세스를 따르고 있습니다. 향후 강의에서는 원시 함수와 계산 그래프 함수에 대한 더 많은 종류의 변환을 소개하겠습니다. 좋은 MLC 프로세스는 이러한 변환들을 함께 구성하여 최종 배포 형태를 형성합니다.

![](../img/mlc_process.png)

## 요약

- 확률적 변환은 가능한 프로그램의 검색 공간을 지정하는 데 도움을 줍니다.
- MetaSchedule은 검색 공간을 검색하고 최적화된 것을 찾습니다.
- 다른 변환을 사용하여 원시 텐서 함수를 최적화된 것으로 교체하고 업데이트된 엔드투엔드 실행 플로우를 만들 수 있습니다.
