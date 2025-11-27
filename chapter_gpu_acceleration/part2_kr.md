## Part 2

이전 장들에서는 CPU 및 GPU 환경을 위한 MLC 플로우 구축에 대해 논의했습니다. 이 장에서는 특화된 하드웨어 백엔드를 위한 개념적 프로그래밍 모델을 구축하는 방법에 초점을 맞춥니다.

### 준비

먼저 필요한 의존성을 가져오겠습니다.

```{.python .input}
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T, relax as R
from tvm import relax
import numpy as np
```

### 하드웨어 특화 트렌드

![](../img/hardware_specialization.png)

머신러닝 하드웨어 환경을 살펴보면 최근 부상하는 주제 중 하나는 특화입니다. 전통적으로 우리는 한 번에 하나의 부동소수점 연산을 수행할 수 있는 범용 스칼라 프로세서에서 솔루션을 구축했습니다. AVX 및 ARM/Neon과 같은 벡터 명령어 세트는 프로그램을 가속화하는 효과적인 방법을 제공하지만 프로그램 작성 방식에 일부 복잡성을 가져옵니다.

머신러닝을 위한 최신 가속기는 다차원 데이터 복사 및 행렬/텐서 계산을 위한 명령어와 함께 텐서 컴퓨팅을 위한 특화된 유닛을 도입했습니다.

#### 특화 코드의 핵심 요소

특화 하드웨어 프로그래밍의 요소를 더 잘 이해하기 위해 먼저 다음 **저수준 NumPy** 코드를 학습하겠습니다. 이 코드는 여전히 Python에서 실행되지만 특화된 하드웨어 백엔드에서 발생할 수 있는 일련의 가능한 연산과 유사합니다.

```{.python .input}
def accel_fill_zero(C):
    C[:] = 0

def accel_tmm_add(C, A, B):
    C[:] += A @ B.T

def accel_dma_copy(reg, dram):
    reg[:] = dram[:]

def lnumpy_tmm(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    # a special accumulator memory
    C_accumulator = np.empty((16, 16), dtype="float32")
    A_reg = np.empty((16, 16), dtype="float32")
    B_reg = np.empty((16, 16), dtype="float32")

    for i in range(64):
        for j in range(64):
            accel_fill_zero(C_accumulator[:,:])
            for k in range(64):
                accel_dma_copy(A_reg[:], A[i * 16 : i * 16 + 16, k * 16 : k * 16 + 16])
                accel_dma_copy(B_reg[:], B[j * 16 : j * 16 + 16, k * 16 : k * 16 + 16])
                accel_tmm_add(C_accumulator[:,:], A_reg, B_reg)
            accel_dma_copy(C[i * 16 : i * 16 + 16, j * 16 : j * 16 + 16], C_accumulator[:,:])
```

![](../img/hardware_specialization_abc.png)

위의 저수준 NumPy 프로그램은 다음과 같은 핵심 요소를 포함합니다:

- 계산의 기본 단위는 16x16x16 행렬 곱셈(`accel_tmm_add`)입니다.
- `accel_tmm_add`는 두 개의 입력 -- `A_reg`와 `B_reg`를 받아 누적기 메모리에 누적합니다.
- 데이터 복사는 특수 함수(`accel_dma_copy`)를 사용하여 수행됩니다.

실제 하드웨어 백엔드에서는 일반적으로 `A_reg`, `B_reg`, `C_accumulator`가 하드웨어의 특수 메모리 영역(또는 레지스터)에 매핑되기를 기대합니다. 이를 **특수 메모리 스코프**라고 합니다. 또한 이러한 설정에서 수행할 수 있는 하드웨어 가속 연산은 제한된 세트입니다. `accel_tmm_add`와 같은 연산은 실제 하드웨어 명령어나 벤더가 제공하는 효율적인 커널 함수 구현에 매핑될 수 있습니다.

다음 코드 블록을 실행하여 저수준 NumPy 코드가 올바르게 실행되는지 확인할 수 있습니다.

```{.python .input}
dtype = "float32"
a_np = np.random.rand(1024, 1024).astype(dtype)
b_np = np.random.rand(1024, 1024).astype(dtype)
c_tmm = a_np @ b_np.T
```

```{.python .input}
c_np = np.empty((1024, 1024), dtype="float32")
lnumpy_tmm(a_np, b_np, c_np)
np.testing.assert_allclose(c_np, c_tmm, rtol=1e-5)
```

#### 텐서화된 계산을 가진 블록

우리의 핵심 관찰 중 하나는 특화된 가속기 코드가 스칼라 계산 단위로 구조화되지 않는다는 것입니다. 지금까지 실행한 대부분의 TensorIR 코드에는 출력 텐서의 단일 요소를 계산하는 블록이 포함되어 있습니다. 많은 특화된 가속기는 텐서의 영역에 대해 계산을 실행합니다. TensorIR의 블록 구조는 이러한 관련 계산을 그룹화하는 데 도움이 됩니다.

```{.python .input}
@tvm.script.ir_module
class MatmulBlockModule:
    @T.prim_func
    def main(
        A: T.Buffer((1024, 1024), "float32"),
        B: T.Buffer((1024, 1024), "float32"),
        C: T.Buffer((1024, 1024), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i0, j0, k0 in T.grid(64, 64, 64):
            with T.block("tmm-16x16"):
                vi0, vj0, vk0 = T.axis.remap("SSR", [i0, j0, k0])
                with T.init():
                    for i1, j1 in T.grid(16, 16):
                        with T.block("tmm_init"):
                            vi1, vj1 = T.axis.remap("SS", [i1, j1])
                            C[vi0 * 16 + vi1, vj0 * 16 + vj1] = T.float32(0)
                
                for i1, j1, k1 in T.grid(16, 16, 16):
                    with T.block("tmm"):
                        vi1, vj1, vk1 = T.axis.remap("SSR", [i1, j1, k1])
                        C[vi0 *16 + vi1, vj0 * 16 + vj1] += \
                            A[vi0 * 16 + vi1, vk0 * 16 + vk1] * B[vj0 * 16 + vj1, vk0 * 16 + vk1]
```

```{.python .input}
MatmulBlockModule.show()
```

Let us take a closer look at the following block

```python
with T.block("tmm-16x16"):
    T.reads(A[vi0 * 16 : vi0 * 16 + 16, vk0 * 16 : vk0 * 16 + 16], B[vj0 * 16 : vj0 * 16 + 16, vk0 * 16 : vk0 * 16 + 16])
    T.writes(C[vi0 * 16 : vi0 * 16 + 16, vj0 * 16 : vj0 * 16 + 16])
    ...
```

이 블록은 `A`와 `B`의 16x16 영역에서 읽고 `C`의 16x16 영역에 씁니다. 이 경우 블록의 내용에는 하위 영역 계산의 특정 구현에 대한 추가 세부 정보가 포함됩니다. 우리는 이 블록을 텐서의 하위 영역에 걸쳐 있는 계산을 포함하므로 **텐서화된 블록**이라고 부릅니다.

다음 코드를 실행하여 TensorIR 모듈이 올바른 결과를 생성하는지 확인할 수 있습니다.

```{.python .input}
a_nd = tvm.nd.array(a_np)
b_nd = tvm.nd.array(b_np)

c_nd = tvm.nd.empty((1024, 1024), dtype="float32")

lib = tvm.build(MatmulBlockModule, target="llvm")
lib["main"](a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_nd.numpy(), c_tmm, rtol=1e-5)
```

#### 텐서화된 블록 주변 루프 변환

여기서 할 수 있는 한 가지는 텐서 계산 블록을 둘러싼 루프를 변환하는 것입니다. 이러한 루프 변환은 다양한 텐서 프로그램 변형의 공간을 활성화하기 위해 주변 반복을 재구성하는 데 도움이 됩니다.

```{.python .input}
sch = tvm.tir.Schedule(MatmulBlockModule)

block_mm = sch.get_block("tmm-16x16")
i, j, k = sch.get_loops(block_mm)

i0, i1 = sch.split(i, [None, 4])

sch.reorder(i0, j, i1, k)
sch.mod.show()
```

#### Blockization -- 텐서화된 블록 생성

대부분의 설정에서 스칼라 계산이 포함된 루프로 시작합니다. TensorIR은 루프의 하위 영역을 함께 그룹화하여 텐서화된 계산 블록을 형성하는 blockization이라는 기본 요소를 제공합니다.

```{.python .input}
@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(
        A: T.Buffer((1024, 1024), "float32"),
        B: T.Buffer((1024, 1024), "float32"),
        C: T.Buffer((1024, 1024), "float32"),
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] += A[vi, vk] * B[vj, vk]
```

```{.python .input}
sch = tvm.tir.Schedule(MatmulModule)
i, j, k = sch.get_loops("matmul")
i, ii = sch.split(i, factors=[None, 16])
j, ji = sch.split(j, factors=[None, 16])
k, ki = sch.split(k, factors=[None, 16])
sch.reorder(i, j, k, ii, ji, ki)
sch.mod.show()
```

```{.python .input}
block_mm = sch.blockize(ii)
sch.mod.show()
```

#### 특수 메모리 스코프를 도입하기 위한 TensorIR 변환

저수준 NumPy 코드에서 언급한 바와 같이, 저수준 TensorIR의 한 가지 핵심 요소는 가속 중에 사용되는 특수 메모리 스코프입니다.

cache_read와 write를 사용하여 중간 메모리 단계를 생성할 수 있습니다.

```{.python .input}
A_reg = sch.cache_read(block_mm, 0, storage_scope="global.A_reg")
B_reg = sch.cache_read(block_mm, 1, storage_scope="global.B_reg")
sch.compute_at(A_reg, k)
sch.compute_at(B_reg, k)

write_back_block = sch.cache_write(block_mm, 0, storage_scope="global.accumulator")
sch.reverse_compute_at(write_back_block, j)
sch.mod.show()
```

![](../img/hardware_specialization_abc.png)

여기서 `global.A_reg`는 두 부분으로 구성됩니다. `global`은 모든 스레드가 메모리에 전역적으로 접근할 수 있음을 나타내며, `A_reg`는 메모리의 **스코프 태그**로 후속 컴파일에서 레지스터와 같은 특수 영역에 매핑할 수 있는 기회를 제공합니다.

### Tensorization

이제 TensorIR의 해당 계산 단계에 매핑되는 블록 세트를 생성했습니다. 남은 단계는 텐서화된 블록 중 일부를 하드웨어 가속 명령어에 매핑되는 특정 구현을 사용하도록 매핑하는 것입니다. 이 매핑 프로세스를 **tensorization**이라고 합니다.

tensorization을 준비하기 위해 먼저 계산 설명과 구현을 포함하는 텐서 내장 함수(TensorIntrin)를 등록합니다.

시스템은 설명을 사용하여 계산과 일치하는 관련 영역을 찾고, 구현은 계산을 가속화된 하드웨어 명령어에 매핑합니다.

```{.python .input}
@T.prim_func
def tmm16_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32", offset_factor=16, scope="global.A_reg")
    B = T.match_buffer(b, (16, 16), "float32", offset_factor=16, scope="global.B_reg")
    C = T.match_buffer(c, (16, 16), "float32", offset_factor=16,  scope="global.accumulator")

    with T.block("root"):
        T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for i, j, k in T.grid(16, 16, 16):
            with T.block(""):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]


@T.prim_func
def tmm16_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    sa = T.int32()
    sb = T.int32()
    sc = T.int32()
    A = T.match_buffer(a, (16, 16), "float32", offset_factor=16, strides=[sa, 1], scope="global.A_reg")
    B = T.match_buffer(b, (16, 16), "float32", offset_factor=16, strides=[sb, 1], scope="global.B_reg")
    C = T.match_buffer(c, (16, 16), "float32", offset_factor=16, strides=[sc, 1], scope="global.accumulator")

    with T.block("root"):
        T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.call_extern(
                "tmm16",
                C.access_ptr("w"),
                A.access_ptr("r"),
                B.access_ptr("r"),
                sa,
                sb,
                sc,
                dtype="int32",
            )
        )

tvm.tir.TensorIntrin.register("tmm16", tmm16_desc, tmm16_impl)
```

준비 단계로 먼저 축소를 초기화 블록과 업데이트 단계로 분해합니다.

```{.python .input}
sch.decompose_reduction(block_mm, k)
sch.mod.show()
```

그런 다음 tensorize를 호출하여 `block_mm`(`matmul_o_update` 블록에 해당)을 `tmm16`의 구현을 사용하도록 매핑할 수 있습니다.

```{.python .input}
sch.tensorize(block_mm, "tmm16")
```

```{.python .input}
sch.mod.show()
```

여기서 `T.call_extern`을 사용하여 환경 내부의 외부 함수를 호출합니다. 다운스트림 컴파일 단계는 구현을 연산을 구현하는 명령어에 쉽게 매핑할 수 있습니다.

또는 tmm16을 이 텐서화된 계산을 구현하는 마이크로 커널에 매핑할 수 있습니다. 다음 코드는 외부 "C" 코드를 통해 이를 수행하는 방법을 보여줍니다(필요한 경우 인라인 어셈블리를 추가로 삽입할 수 있습니다).

```{.python .input}
def tmm_kernel():
    cc_code = """
      extern "C" int tmm16(float *cc, float *aa, float *bb, int stride_a, int stride_b, int stride_c) {
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                for (int k = 0; k < 16; ++k) {
                    cc[i * stride_c + j] += aa[i * stride_a + k] * bb[j * stride_b + k];
                }
            }
        }
        return 0;
      }
    """
    from tvm.contrib import utils, clang

    temp = utils.tempdir()
    ll_path = temp.relpath("temp.ll")
    # Create LLVM ir from c source code
    ll_code = clang.create_llvm(cc_code, output=ll_path)
    return ll_code

sch.annotate(i, "pragma_import_llvm", tmm_kernel())
```

We can then go and execute the following code-block, which redirects the tensorized computation to the custom defined `tmm_kernel`.

```
<!-- todo -->
<!-- For CI, do not run this part of the code -->
a_nd = tvm.nd.array(a_np)
b_nd = tvm.nd.array(b_np)

c_nd = tvm.nd.empty((1024, 1024), dtype="float32")

lib = tvm.build(sch.mod, target="llvm")
lib["main"](a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_nd.numpy(), c_tmm, rtol=1e-5)
```

### 논의

이 섹션에서는 특화된 하드웨어 지원의 핵심 요소 세트를 다룹니다. 여기서 핵심 구조 중 하나는 텐서화된 블록과 텐서 하위 영역과 함께하는 계산입니다. TensorIR에는 기초 요소를 기반으로 구축되는 추가 속성도 포함됩니다:

- 특화된 메모리의 레이아웃 제약 조건
- 스레드 계층 구조와의 상호 작용

한 강의에서 이를 모두 다룰 충분한 시간이 없지만 일부 추가 콘텐츠에 대한 선택적 읽기 자료를 추가할 예정입니다.

### 요약

- 텐서화된 계산을 향한 하드웨어 특화의 전반적인 트렌드
- 텐서화된 블록을 사용한 TensorIR 변환
- Tensorization: 루프 계산의 블록을 특화된 구현에 매핑하는 프로세스
