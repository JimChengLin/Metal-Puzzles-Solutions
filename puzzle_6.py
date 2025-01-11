import mlx.core as mx
from utils import MetalKernel, MetalProblem

def map_spec(a: mx.array):
    return a + 10

def map_threadgroup_test(a: mx.array):
    source = """
        uint i = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
        // FILL ME IN (roughly 1-3 lines)
        if (i < a_shape[0]) {
            out[i] = a[i] + 10;
        }
    """

    kernel = MetalKernel(
        name="threadgroups",
        input_names=["a"],
        output_names=["out"],
        source=source,
    )

    return kernel

SIZE = 9
a = mx.arange(SIZE)
output_shape = (SIZE,)

problem = MetalProblem(
    "Threadgroups",
    map_threadgroup_test,
    [a],
    output_shape,
    grid=(12,1,1),
    threadgroup=(4,1,1),
    spec=map_spec
)

graph = problem.show()
graph.render("dump.png", height=1000)
problem.check()