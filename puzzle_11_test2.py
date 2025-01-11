import mlx.core as mx
from utils import MetalKernel, MetalProblem


def conv_spec(a: mx.array, b: mx.array):
    out = mx.zeros(*a.shape)
    len = b.shape[0]
    for i in range(a.shape[0]):
        out[i] = sum([a[i + j] * b[j] for j in range(len) if i + j < a.shape[0]])
    return out


def conv_test(a: mx.array, b: mx.array):
    header = """
        constant uint THREADGROUP_MAX_CONV_SIZE = 11;
        constant uint MAX_CONV = 4;
    """

    source = """
        uint i = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
        uint local_i = thread_position_in_threadgroup.x;
        
        threadgroup float shared_input[THREADGROUP_MAX_CONV_SIZE];
        shared_input[local_i] = 0;
        if (local_i >= threads_per_threadgroup.x - 3) {
            shared_input[local_i + 3] = 0;
        }

        if (i < a_shape[0]) {
            threadgroup float shared_conv[MAX_CONV];
            
            shared_input[local_i] = a[i];
            
            if (local_i < b_shape[0]) {
                 shared_conv[local_i] = b[local_i];
            }
 
            threadgroup_barrier(mem_flags::mem_threadgroup);
             
            out[i] = shared_input[local_i] * shared_conv[0] + shared_input[local_i + 1] * shared_conv[1] + shared_input[local_i + 2] * shared_conv[2] + shared_input[local_i + 3] * shared_conv[3];
        }
    """

    kernel = MetalKernel(
        name="1D_conv",
        input_names=["a", "b"],
        output_names=["out"],
        header=header,
        source=source,
    )

    return kernel


# Test 2
a = mx.arange(15, dtype=mx.float32)
b = mx.arange(4, dtype=mx.float32)
output_shape = (15,)

problem = MetalProblem(
    "1D Conv (Full)",
    conv_test,
    [a, b],
    output_shape,
    grid=(16,1,1),
    threadgroup=(8,1,1),
    spec=conv_spec
)

graph = problem.show()
graph.render("dump.png", height=1000)
problem.check()