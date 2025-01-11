import mlx.core as mx
from utils import MetalKernel, MetalProblem

def map_spec(a: mx.array):
    return a + 10

def shared_test(a: mx.array):
    header = """
        constant uint THREADGROUP_MEM_SIZE = 4;
    """

    source = """
        threadgroup float shared[THREADGROUP_MEM_SIZE];
        uint i = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
        uint local_i = thread_position_in_threadgroup.x;

        // if (i < a_shape[0]) {
            shared[local_i] = a[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        // }

        out[i] = shared[local_i] + 10;
    """

    kernel = MetalKernel(
        name="threadgroup_memory",
        input_names=["a"],
        output_names=["out"],
        header=header,
        source=source,
    )

    return kernel

SIZE = 8
a = mx.ones(SIZE)
output_shape = (SIZE,)

problem = MetalProblem(
    "Threadgroup Memory",
    shared_test,
    [a],
    output_shape,
    grid=(SIZE,1,1),
    threadgroup=(4,1,1),
    spec=map_spec
)

graph = problem.show()
graph.render("dump.png", height=1000)
problem.check()