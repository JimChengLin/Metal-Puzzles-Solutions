import mlx.core as mx
from utils import MetalKernel, MetalProblem

def map_spec(a: mx.array):
    return a + 10

def map_test(a: mx.array):
    source = """
        uint local_i = thread_position_in_grid.x;
        out[local_i] = a[local_i] + 10;
    """

    kernel = MetalKernel(
        name="map",
        input_names=["a"],
        output_names=["out"],
        source=source,
    )

    return kernel

SIZE = 4
a = mx.arange(SIZE)
output_shape = (SIZE,)

problem = MetalProblem(
    "Map",
    map_test,
    [a],
    output_shape,
    grid=(SIZE,1,1),
    spec=map_spec
)
graph = problem.show()
graph.render("dump.png", height=1000)
problem.check()