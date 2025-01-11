import mlx.core as mx
from utils import MetalKernel, MetalProblem

def dot_spec(a: mx.array, b: mx.array):
    return a @ b

def dot_test(a: mx.array, b: mx.array):
    header = """
        constant uint THREADGROUP_MEM_SIZE = 8;
    """

    source = """
        threadgroup float shared[THREADGROUP_MEM_SIZE];
        uint i = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
        uint local_i = thread_position_in_threadgroup.x;
        // FILL ME IN (roughly 11 lines)
        
        shared[local_i] = a[i] * b[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (i == 0) {
            out[0] = shared[0] + shared[1] + shared[2] + shared[3] + shared[4] + shared[5] + shared[6] + shared[7];
        }
    """

    kernel = MetalKernel(
        name="dot_product",
        input_names=["a", "b"],
        output_names=["out"],
        header=header,
        source=source,
    )

    return kernel

SIZE = 8
a = mx.arange(SIZE, dtype=mx.float32)
b = mx.arange(SIZE, dtype=mx.float32)
output_shape = (1,)

problem = MetalProblem(
    "Dot Product",
    dot_test,
    [a, b],
    output_shape,
    grid=(SIZE,1,1),
    threadgroup=(SIZE,1,1),
    spec=dot_spec
)

graph = problem.show()
graph.render("dump.png", height=1000)
problem.check()