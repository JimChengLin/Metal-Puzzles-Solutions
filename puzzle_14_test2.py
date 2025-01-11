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
        
        uint sum = 0;
        for (uint k = 0; k < 3; k += 1) {
            uint a_idx = j * a_shape[1] + 3 * k + local_i;
            uint b_idx = (3 * k + local_j) * a_shape[1] + i;
            if (j < a_shape[0] && 3 * k + local_i < a_shape[1]) {
                a_shared[local_j][local_i] = a[a_idx];
            } else {
                a_shared[local_j][local_i] = 0;
            }
            if ((3 * k + local_j) < a_shape[0] && i < a_shape[1]) {
                b_shared[local_j][local_i] = b[b_idx];
            } else {
                b_shared[local_j][local_i] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            sum = sum + a_shared[local_j][0] * b_shared[0][local_i] + a_shared[local_j][1] * b_shared[1][local_i] + a_shared[local_j][2] * b_shared[2][local_i];
        }
        
        if (j < a_shape[0] && i < a_shape[1]) {
            out[j * a_shape[1] + i] = sum;
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

# Test 2
SIZE = 8
a = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE))
b = mx.arange(SIZE * SIZE, dtype=mx.float32).reshape((SIZE, SIZE)).T
output_shape = (SIZE, SIZE)

problem = MetalProblem(
    "Matmul (Full)",
    matmul_test,
    [a, b],
    output_shape,
    grid=(9,9,1),
    threadgroup=(3,3,1),
    spec=matmul_spec
)

graph = problem.show()
graph.render("dump.png", height=4000)
problem.check()
