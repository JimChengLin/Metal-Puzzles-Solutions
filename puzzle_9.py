import mlx.core as mx
from utils import MetalKernel, MetalProblem

def pooling_spec(a: mx.array):
    out = mx.zeros(*a.shape)
    for i in range(a.shape[0]):
        out[i] = a[max(i - 2, 0) : i + 1].sum()
    return out

def pooling_test(a: mx.array):
    header = """
        constant uint THREADGROUP_MEM_SIZE = 8;
    """

    source = """
        threadgroup float shared[THREADGROUP_MEM_SIZE];
        uint i = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
        uint local_i = thread_position_in_threadgroup.x;
        
        shared[local_i] = a[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (i >= 2) {
            out[i] = shared[i-2] + shared[i-1] + shared[i];
        } else if (i == 1) {
            out[i] = shared[i-1] + shared[i];
        } else if (i == 0) {
            out[i] = shared[i];
        }
    """

    kernel = MetalKernel(
        name="pooling",
        input_names=["a"],
        output_names=["out"],
        header=header,
        source=source,
    )

    return kernel

SIZE = 8
a = mx.arange(SIZE)
output_shape = (SIZE,)

problem = MetalProblem(
    "Pooling",
    pooling_test,
    [a],
    output_shape,
    grid=(SIZE,1,1),
    threadgroup=(SIZE,1,1),
    spec=pooling_spec
)

graph = problem.show()
graph.render("dump.png", height=1000)
problem.check()