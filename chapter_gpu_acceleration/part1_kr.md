## Part 1

이전 장에서는 CPU 환경에서의 MLC 플로우에 대해 논의했습니다. 이 장에서는 일부 최적화를 GPU로 가져오는 방법에 대해 논의하겠습니다. CUDA 용어를 사용할 것입니다. 그러나 동일한 개념 세트가 다른 종류의 GPU에도 적용됩니다.

### 패키지 설치

이 과정에서는 오픈소스 머신러닝 컴파일 프레임워크인 TVM의 진행 중인 개발을 사용할 것입니다. MLC 과정을 위한 패키지 버전을 설치하는 다음 명령을 제공합니다. **part 1**의 특정 노트북은 CUDA 11 환경에 의존합니다.

```bash
python3 -m pip install mlc-ai-nightly-cu110 -f https://mlc.ai/wheels
```

**참고: 빌드 시스템이 아직 GPU 지원이 없으므로 일부 코드는 평가되지 않습니다.**

### 준비

먼저 필요한 의존성을 가져오겠습니다.

```{.python .input}
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T, relax as R
from tvm import relax
import numpy as np
```

### GPU 아키텍처

GPU 아키텍처가 어떻게 생겼는지 검토하는 것으로 시작하겠습니다. 일반적인 GPU는 스트림 멀티프로세서 모음을 포함하며, 각 멀티프로세서는 많은 코어를 가지고 있습니다. GPU 장치는 대규모 병렬이며 많은 작업을 동시에 실행할 수 있습니다.

![](../img/gpu_arch.png)

GPU를 프로그래밍하려면 스레드 블록 세트를 생성해야 하며, 각 스레드는 코어에 매핑되고 스레드 블록은 스트림 멀티프로세서에 매핑됩니다.

![](../img/gpu_stream_processors.png)

벡터 덧셈 예제를 사용하여 GPU 프로그래밍을 시작하겠습니다. 다음 TensorIR 프로그램은 두 벡터 A와 B를 받아 요소별 덧셈을 수행하고 결과를 C에 저장합니다.

```{.python .input}
@tvm.script.ir_module
class MyModuleVecAdd:
    @T.prim_func
    def main(A: T.Buffer((1024,), "float32"),
             B: T.Buffer((1024,), "float32"),
             C: T.Buffer((1024,), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in T.grid(1024):
            with T.block("C"):
                vi = T.axis.remap("S", [i])
                C[vi] = A[vi] + B[vi]
```

먼저 루프 `i`를 두 개의 루프로 분할합니다.

```{.python .input}
sch = tvm.tir.Schedule(MyModuleVecAdd)
block_C = sch.get_block("C")
i, = sch.get_loops(block=block_C)
i0, i1 = sch.split(i, [None, 128])
sch.mod.show()
```

#### GPU 스레드 블록

그런 다음 반복자를 GPU 스레드 블록에 바인딩합니다. 각 스레드는 두 개의 인덱스 -- `threadIdx.x`와 `blockIdx.x`로 매개변수화됩니다. 실제로는 다차원 스레드 인덱스를 가질 수 있지만, 한 차원으로 간단하게 유지합니다.

![](../img/gpu_thread_blocks.png)

```{.python .input}
sch.bind(i0, "blockIdx.x")
sch.bind(i1, "threadIdx.x")
sch.mod.show()
```

#### GPU에서 TensorIR 함수 빌드 및 실행

GPU에서 결과 함수를 빌드하고 테스트할 수 있습니다.

```python
rt_mod = tvm.build(sch.mod, target="cuda")

A_np = np.random.uniform(size=(1024,)).astype("float32")
B_np = np.random.uniform(size=(1024,)).astype("float32")
A_nd = tvm.nd.array(A_np, tvm.cuda(0))
B_nd = tvm.nd.array(B_np, tvm.cuda(0))
C_nd = tvm.nd.array(np.zeros((1024,), dtype="float32"), tvm.cuda(0))

rt_mod["main"](A_nd, B_nd, C_nd)
print(A_nd)
print(B_nd)
print(C_nd)
```

### 윈도우 합 예제

이제 다른 예제인 윈도우 합으로 넘어가겠습니다. 이 프로그램은 미리 정의된 가중치 `[1,1,1]`을 가진 "합성곱"의 기본 버전으로 볼 수 있습니다. 입력을 슬라이딩하면서 세 개의 인접한 값을 더합니다.

![](../img/window_sum.png)

```{.python .input}
@tvm.script.ir_module
class MyModuleWindowSum:
    @T.prim_func
    def main(A: T.Buffer[(1027,), "float32"],
             B: T.Buffer[(1024,), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in T.grid(1024):
            with T.block("C"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi] + A[vi + 1] + A[vi + 2]
```

먼저 루프를 GPU 스레드에 바인딩할 수 있습니다.

```{.python .input}
sch = tvm.tir.Schedule(MyModuleWindowSum)
nthread = 128
block_C = sch.get_block("C")
i,  = sch.get_loops(block=block_C)
i0, i1 = sch.split(i, [None, nthread])
sch.bind(i0, "blockIdx.x")
sch.bind(i1, "threadIdx.x")
sch.mod.show()
```

![](../img/gpu_stream_processors.png)

중요한 점은 이 경우 재사용 기회가 있다는 것입니다. 각 GPU 스레드 블록에는 블록 내 모든 스레드가 접근할 수 있는 공유 메모리가 포함되어 있음을 기억하십시오. `cache_read`를 사용하여 세그먼트(아래 녹색)를 공유 메모리에 캐시하는 중간 단계를 추가합니다. 캐싱이 완료되면 스레드가 공유 메모리에서 읽을 수 있습니다.

```{.python .input}
A_shared = sch.cache_read(block_C, read_buffer_index=0, storage_scope="shared")
sch.compute_at(A_shared, i1)
sch.mod.show()
```

메모리가 스레드 간에 공유되므로 루프를 다시 분할하고 페칭 프로세스의 내부 반복자를 스레드 인덱스에 바인딩해야 합니다. 이 기법을 **협력 페칭**이라고 하며, 여러 스레드가 함께 작업하여 데이터를 공유 메모리로 가져옵니다. 다음 읽기 프로세스는 다를 수 있습니다.

```{.python .input}
ax = sch.get_loops(A_shared)[-1]
ax0, ax1 = sch.split(ax, [None, nthread])
sch.bind(ax1, "threadIdx.x")
sch.mod.show()
```

해당 저수준 코드(CUDA)를 검사할 수 있습니다. 생성된 코드에는 두 부분이 포함됩니다:

- GPU 드라이버를 호출하는 호스트 부분
- 해당 계산을 실행하는 cuda 커널

다음 코드를 사용하여 cuda 커널을 출력할 수 있습니다. 프로그램을 실행하려면 호스트와 커널 코드가 모두 필요하므로 최종 코드 생성 결과를 검사하는 빠른 방법일 뿐입니다.

특히 빌드 프로세스는 스레드 블록 내에서 사용되는 최소 영역을 사용하도록 공유 메모리 단계를 자동으로 압축합니다.

```python
rt_mod = tvm.build(sch.mod, target="cuda")
print(rt_mod.imported_modules[0].get_source())
```

#### 다른 GPU 플랫폼용 코드 빌드

MLC 프로세스는 일반적으로 여러 종류의 하드웨어 플랫폼을 대상으로 지원하며, 대상 매개변수를 변경하여 Metal 코드(다른 종류의 GPU 프로그래밍 모델)를 생성할 수 있습니다.

```python
rt_mod = tvm.build(sch.mod, target="metal")
print(rt_mod.imported_modules[0].get_source())
```

### 행렬 곱셈

이제 약간 더 복잡한 것으로 넘어가서 GPU에서 행렬 곱셈 최적화를 시도해보겠습니다. GPU 성능 최적화를 위한 두 가지 일반적인 기법을 살펴보겠습니다.

```{.python .input}
@tvm.script.ir_module
class MyModuleMatmul:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"),
             B: T.Buffer((1024, 1024), "float32"),
             C: T.Buffer((1024, 1024), "float32")) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

#### 로컬 블로킹

![](../img/gpu_local_blocking.png)

전체 메모리 재사용을 늘리기 위해 루프를 타일링할 수 있습니다. 특히 A와 B에서 데이터 스트라이프를 한 번만 로드한 다음 이를 사용하여 `V * V` 행렬 곱셈 결과를 수행하도록 로컬 타일을 도입합니다.

이러한 로컬 타일링은 스트라이프의 각 요소가 `V`번 재사용되므로 메모리 압력을 줄이는 데 도움이 됩니다.

```{.python .input}
def blocking(sch,
             tile_local_y,
             tile_local_x,
             tile_block_y,
             tile_block_x,
             tile_k):
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])
    sch.unroll(k1)
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    sch.bind(i1, "threadIdx.y")
    sch.bind(j1, "threadIdx.x")
    sch.decompose_reduction(block_C, k0)

    return sch

sch = tvm.tir.Schedule(MyModuleMatmul)
sch = blocking(sch, 8, 8, 8, 8, 4)
sch.mod.show()
```

```python
rt_mod = tvm.build(sch.mod, target="cuda")
dev = tvm.cuda(0)
A_np = np.random.uniform(size=(1024, 1024)).astype("float32")
B_np = np.random.uniform(size=(1024, 1024)).astype("float32")
A_nd = tvm.nd.array(A_np, dev)
B_nd = tvm.nd.array(B_np, dev)
C_nd = tvm.nd.array(np.zeros((1024, 1024), dtype="float32"), dev)

num_flop = 2 * 1024 * 1024 * 1024
evaluator = rt_mod.time_evaluator("main", dev, number=10)

print("GEMM-Blocking: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))
```

### 공유 메모리 블로킹

![](../img/gpu_shared_blocking.png)

첫 번째 시도에서는 동일한 GPU 스레드 블록에 있는 인접 스레드를 고려하지 않았으며, 그들이 공통으로 필요로 하는 데이터를 공유 메모리에 로드할 수 있습니다.

다음 변환이 이를 수행합니다.

```{.python .input}
def cache_read_and_coop_fetch(sch, block, nthread, read_idx, read_loc):
    read_cache = sch.cache_read(block=block, read_buffer_index=read_idx, storage_scope="shared")
    sch.compute_at(block=read_cache, loop=read_loc)
    # vectorized cooperative fetch
    inner0, inner1 = sch.get_loops(block=read_cache)[-2:]
    inner = sch.fuse(inner0, inner1)
    _, tx, vec = sch.split(loop=inner, factors=[None, nthread, 4])
    sch.vectorize(vec)
    sch.bind(tx, "threadIdx.x")


def blocking_with_shared(
    sch,
    tile_local_y,
    tile_local_x,
    tile_block_y,
    tile_block_x,
    tile_k):
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")

    i, j, k = sch.get_loops(block=block_C)

    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])

    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    tx = sch.fuse(i1, j1)
    sch.bind(tx, "threadIdx.x")
    nthread = tile_block_y * tile_block_x
    cache_read_and_coop_fetch(sch, block_C, nthread, 0, k0)
    cache_read_and_coop_fetch(sch, block_C, nthread, 1, k0)
    sch.decompose_reduction(block_C, k0)

    return sch

sch = tvm.tir.Schedule(MyModuleMatmul)
sch = blocking_with_shared(sch, 8, 8, 8, 8, 8)
sch.mod.show()
```

```python
rt_mod = tvm.build(sch.mod, target="cuda")
dev = tvm.cuda(0)
evaluator = rt_mod.time_evaluator("main", dev, number=10)

print("GEMM-Blocking: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))
```

### 자동 프로그램 최적화 활용

지금까지 GPU에서 TensorIR 프로그램을 최적화하기 위해 수동으로 변환을 작성했습니다. 자동 프로그램 최적화 프레임워크를 활용하여 동일한 프로그램을 튜닝할 수 있습니다. 다음 코드가 이를 수행하며, 여기서는 작은 숫자만 설정했으며 완료하는 데 몇 분 정도 걸릴 수 있습니다.

```python
from tvm import meta_schedule as ms

database = ms.tune_tir(
    mod=MyModuleMatmul,
    target="nvidia/tesla-p100",
    max_trials_global=64,
    num_trials_per_iter=64,
    work_dir="./tune_tmp",
)
sch = ms.tir_integration.compile_tir(database, MyModuleMatmul, "nvidia/tesla-p100")
sch.mod.show()
```

```python
rt_mod = tvm.build(sch.mod, target="nvidia/tesla-p100")
dev = tvm.cuda(0)
evaluator = rt_mod.time_evaluator("main", dev, number=10)

print("MetaSchedule: %f GFLOPS" % (num_flop / evaluator(A_nd, B_nd, C_nd).mean / 1e9))
```

### 요약

이 장에서는 MLC의 또 다른 축인 하드웨어 가속을 위해 프로그램을 변환하는 방법을 학습합니다. MLC 프로세스는 입력 모델을 다양한 GPU 프로그래밍 모델과 환경으로 연결하는 데 도움을 줍니다. 다음 장에서도 더 많은 하드웨어 특화 주제를 다룰 예정입니다.

- 일반적인 GPU는 2레벨 계층 구조를 포함합니다. 각 스레드는 (CUDA 용어로) `threadIdx.x`와 `blockIdx.x`로 인덱싱됩니다(다차원 인덱스도 가능하지만 하나로 융합할 수 있습니다).
- 공유 메모리는 동일한 블록 내 스레드 간에 일반적으로 사용되는 데이터를 캐시하는 데 도움이 됩니다.
- GPU 최적화 중에 메모리 재사용을 장려합니다.
    