import mlx.core as mx
from utils import MetalKernel, MetalProblem

def matmul_spec(a: mx.array, b: mx.array):
    return a @ b

def matmul_test(a: mx.array, b: mx.array):
    header = """
        constant uint THREADGROUP_MEM_SIZE = 3;
    """

    source = """
        threadgroup float a_shared[THREADGROUP_MEM_SIZE][THREADGROUP_MEM_SIZE];
        threadgroup float b_shared[THREADGROUP_MEM_SIZE][THREADGROUP_MEM_SIZE];

        uint i = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
        uint j = threadgroup_position_in_grid.y * threads_per_threadgroup.y + thread_position_in_threadgroup.y;

        uint local_i = thread_position_in_threadgroup.x;
        uint local_j = thread_position_in_threadgroup.y;
        
        a_shared[local_j][local_i] = 0;
        b_shared[local_j][local_i] = 0;
        
        if (j < a_shape[0] && i < a_shape[1]) {
            a_shared[local_j][local_i] = a[j * a_shape[1] + i];
            b_shared[local_j][local_i] = b[j * a_shape[1] + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        if (j < a_shape[0] && i < a_shape[1]) {
            out[j * a_shape[1] + i] = a_shared[local_j][0] * b_shared[0][local_i] + a_shared[local_j][1] * b_shared[1][local_i] + a_shared[local_j][2] * b_shared[2][local_i];
        }
    """

    kernel = MetalKernel(
        name="matmul",
        input_names=["a", "b"],
        output_names=["out"],
        header=header,
        source=source,
    )

    return kernel

# Test 1
SIZE = 2
a = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE))
b = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE)).T
output_shape = (SIZE, SIZE)

problem = MetalProblem(
    "Matmul (Simple)",
    matmul_test,
    [a, b],
    output_shape,
    grid=(3,3,1),
    threadgroup=(3,3,1),
    spec=matmul_spec
)

graph = problem.show()
graph.render("dump.png", height=1000)
problem.check()
