import mlx.core as mx
from utils import MetalKernel, MetalProblem

THREADGROUP_MEM_SIZE = 8
def axis_sum_spec(a: mx.array):
    out = mx.zeros((a.shape[0], (a.shape[1] + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE))
    for j, i in enumerate(range(0, a.shape[-1], THREADGROUP_MEM_SIZE)):
        out[..., j] = a[..., i : i + THREADGROUP_MEM_SIZE].sum(-1)
    return out

def axis_sum_test(a: mx.array):
    header = """
        constant uint THREADGROUP_MEM_SIZE = 8;
    """

    source = """
        threadgroup float cache[THREADGROUP_MEM_SIZE];
        uint i = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
        uint local_i = thread_position_in_threadgroup.x;
        uint batch = threadgroup_position_in_grid.y;
        
        cache[local_i] = 0;
        
        if (local_i < a_shape[1] && batch < a_shape[0]) {
            cache[local_i] = a[batch * a_shape[1] + local_i];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (uint j = 0; j < 3; j += 1) {
            if ((local_i % (2 << j)) == ((2 << j) - 1)) {
                cache[local_i] = cache[local_i] + cache[local_i - (1 << j)];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        if (local_i == threads_per_threadgroup.x -1) {
            out[threadgroup_position_in_grid.y] = cache[local_i];
        }
    """

    kernel = MetalKernel(
        name="axis_sum",
        input_names=["a"],
        output_names=["out"],
        header=header,
        source=source,
    )

    return kernel

BATCH = 4
SIZE = 6
a = mx.arange(BATCH * SIZE).reshape((BATCH, SIZE))
output_shape = (BATCH, 1)

problem = MetalProblem(
    "Axis Sum",
    axis_sum_test,
    [a],
    output_shape,
    grid=(8,BATCH,1),
    threadgroup=(8,1,1),
    spec=axis_sum_spec
)

graph = problem.show()
graph.render("dump.png", height=1000)
problem.check()