import mlx.core as mx
from utils import MetalKernel, MetalProblem

def zip_spec(a: mx.array, b: mx.array):
    return a + b

def broadcast_test(a: mx.array, b: mx.array):
    source = """
        uint thread_x = thread_position_in_grid.x;
        uint thread_y = thread_position_in_grid.y;
        if (thread_x < b_shape[1] && thread_y < a_shape[0]) {
            out[thread_x + thread_y * a_shape[0]] = b[thread_x] + a[thread_y];
        }
    """

    kernel = MetalKernel(
        name="broadcast",
        input_names=["a", "b"],
        output_names=["out"],
        source=source,
    )

    return kernel

SIZE = 2
a = mx.arange(SIZE).reshape(SIZE, 1)
b = mx.arange(SIZE).reshape(1, SIZE)
output_shape = (SIZE,SIZE)

problem = MetalProblem(
    "Broadcast",
    broadcast_test,
    [a, b],
    output_shape,
    grid=(3,3,1),
    spec=zip_spec
)

graph = problem.show()
graph.render("dump.png", height=1000)
problem.check()