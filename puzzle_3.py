import mlx.core as mx
from utils import MetalKernel, MetalProblem

def map_spec(a: mx.array):
    return a + 10

def map_guard_test(a: mx.array):
    source = """
        uint local_i = thread_position_in_grid.x;
        if (local_i < a_shape[0]) {
            out[local_i] = a[local_i] + 10;
        }
    """

    kernel = MetalKernel(
        name="guard",
        input_names=["a"],
        output_names=["out"],
        source=source,
    )

    return kernel

SIZE = 4
a = mx.arange(SIZE)
output_shape = (SIZE,)

problem = MetalProblem(
    "Guard",
    map_guard_test,
    [a],
    output_shape,
    grid=(8,1,1),
    spec=map_spec
)

graph = problem.show()
graph.render("dump.png", height=1000)
problem.check()