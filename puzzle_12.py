import mlx.core as mx
from utils import MetalKernel, MetalProblem

THREADGROUP_MEM_SIZE = 8
def prefix_sum_spec(a: mx.array):
    out = mx.zeros((a.shape[0] + THREADGROUP_MEM_SIZE - 1) // THREADGROUP_MEM_SIZE)
    for j, i in enumerate(range(0, a.shape[-1], THREADGROUP_MEM_SIZE)):
        out[j] = a[i : i + THREADGROUP_MEM_SIZE].sum()
    return out

def prefix_sum_test(a: mx.array):
    header = """
        constant uint THREADGROUP_MEM_SIZE = 8;
    """

    source = """
        threadgroup float cache[THREADGROUP_MEM_SIZE];
        uint i = threadgroup_position_in_grid.x * threads_per_threadgroup.x + thread_position_in_threadgroup.x;
        uint local_i = thread_position_in_threadgroup.x;
        
        cache[local_i] = 0;
        
        if (i < a_shape[0]) {
            cache[local_i] = a[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // up
            if (local_i % 2 == 1) {
                cache[local_i] = cache[local_i - 1] + cache[local_i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (local_i % 4 == 3) {
                cache[local_i] = cache[local_i - 2] + cache[local_i];
            }
            
            // down
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (local_i == 7) {
               cache[7] = cache[3];
               cache[3] = 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (local_i % 4 == 3) {
               uint tmp = cache[local_i];
               cache[local_i] = cache[local_i] + cache[local_i - 2];
               cache[local_i - 2] = tmp;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (local_i % 2 == 1) {
               uint tmp = cache[local_i];
               cache[local_i] = cache[local_i] + cache[local_i - 1];
               cache[local_i - 1] = tmp;
            }
        }
        
        if (local_i == 7) {
            if (i < a_shape[0]) {
                out[threadgroup_position_in_grid.x] = cache[local_i] + a[i];
            } else {
                out[threadgroup_position_in_grid.x] = cache[local_i];
            }
        }
    """

    kernel = MetalKernel(
        name="prefix_sum",
        input_names=["a"],
        output_names=["out"],
        header=header,
        source=source,
    )

    return kernel

# Test 1
SIZE = 8
a = mx.arange(SIZE)
output_shape = (1,)

problem = MetalProblem(
    "Prefix Sum (Simple)",
    prefix_sum_test,
    [a],
    output_shape,
    grid=(8,1,1),
    threadgroup=(8,1,1),
    spec=prefix_sum_spec
)


graph = problem.show()
graph.render("dump.png", height=1000)
problem.check()