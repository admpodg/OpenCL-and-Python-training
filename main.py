from time import time  # Import time tools

import pyopencl as cl
import numpy as np
import deviceInfoPyopencl as device_info
import numpy.linalg as la

# input vectors
a = np.random.rand(281600000).astype(np.float32)
b = np.random.rand(281600000).astype(np.float32)




def test_cpu_vector_using_numpy(a, b):

    c_cpu = np.empty_like(a)
    cpu_start_time = time()
    c_cpu = np.cos(a)
    cpu_end_time = time()
    print("CPU numpy time: {0} s".format(cpu_end_time - cpu_start_time))
    return c_cpu


def test_gpu_vector_sum(a, b):


    # define the PyOpenCL Context
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context, \
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    # prepare the data structure
    a_buffer = cl.Buffer \
        (context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
    b_buffer = cl.Buffer \
        (context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
    c_buffer = cl.Buffer \
        (context, cl.mem_flags.WRITE_ONLY, b.nbytes)
    program = cl.Program(context, """
    __kernel void sum(__global const float *a, __global float *c)
    {
        int i = get_global_id(0);
        
            c[i] = cos(a[i]);
        
    }""").build()
    # start the gpu test
    c_gpu = np.empty_like(a)

    gpu_start_time = time()
    event = program.sum(queue, a.shape, None, a_buffer, c_buffer)
    event.wait()
    elapsed = 1e-9 * (event.profile.end - event.profile.start)
    print("GPU Kernel evaluation Time: {0} s".format(elapsed))
    cl._enqueue_read_buffer(queue, c_buffer, c_gpu).wait()
    gpu_end_time = time()
    print("GPU Time: {0} s".format(gpu_end_time - gpu_start_time))
    return c_gpu


# start the test
if __name__ == "__main__":
    # print the device info
    device_info.print_device_info()
    # call the test on the cpu
   # cpu_result = test_cpu_vector_sum(a, b)
    # call the test on the gpu
    gpu_result = test_gpu_vector_sum(a, b)

    # numpy
    numpy_result = test_cpu_vector_using_numpy(a, b)

    print("arrays are equal: %d ", np.array_equal(gpu_result, numpy_result))
    print("gpu result {0}".format( gpu_result[gpu_result.size-10:gpu_result.size-1]))
    print("cpu result {0}".format(numpy_result[numpy_result.size-10:numpy_result.size-1]))


    #assert (la.norm(cpu_result - gpu_result)) < 1e-5
