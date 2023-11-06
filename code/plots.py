# M = N = K
import matplotlib.pyplot as plt

n_runs = 25
matrix_sizez =  [128,    256,    512,     1024,     2048,     4096,    8128]
register_tiled =[99.77,  552.61, 1945.18, 5839.36,  6701.46,  9989.81, 12714.6]
tensor_naive =  [135.65, 737.14, 2583.10, 6505.95,  9421.06,  17568.2, 17369.7]
tensor_optim =  [127.41, 700.22, 2401.89, 11623.10, 17121.70, 37243.9, 37713.9]
cublas =        [246.72, 1864.14,14708.8, 46939.50, 121413.0, 236251.0,259219]

def plot_performance():
    n_runs = 25
    matrix_size =  [128,    256,    512,     1024,     2048,     3072,     4096,    5120,    6144,    7168,     8128,   9600]
    register_tiled =[99.77,  552.61, 1945.18, 5839.36,  6701.46,  9027.76, 9989.81, 12948.7, 12456.9, 12738.7, 12714.6, 12815.5]
    tensor_naive =  [135.65, 737.14, 2583.10, 6505.95,  9421.06,  16411.6, 17568.2, 17388.2, 17372.8, 17350.8, 17369.7, 17484.9]
    tensor_optim =  [127.41, 700.22, 2401.89, 11623.10, 17121.70, 34753.1, 37243.9, 36816.2, 35904.1, 35921.4, 37713.9, 35880.5]
    cublas =        [226.72, 1789.57,12341.9, 34087.00, 145284.0, 153493,  214832,  239835,  250294,  244774,  258781,  256417]

    #plt.plot(matrix_size, list(map(lambda _: peak_performance, matrix_size)))
    plt.plot(matrix_size, register_tiled, marker="s")
    plt.plot(matrix_size, tensor_naive, marker="s")
    plt.plot(matrix_size, tensor_optim, marker="s")
    #plt.plot(matrix_size, cublas, marker="s")
    #plt.yscale("log")
    plt.legend(["Cuda Cores", "Tensor Core Naive", "Tensor Core Optimized", "cuBLAS"])
    plt.ylabel("GFlops")
    plt.xlabel("Size WxWxW")
    plt.title("Matrix multiplication on MxNxK matrices")
    plt.savefig("performance_no_cublas.png", dpi=92, bbox_inches="tight")
    plt.show()
    

def plot_optimizations():
    keep_c_no_vec = 37243.9
    cache_c = 5967.14
    global_c = 12668.2
    keep_c_bank_confligt = 19627.9
    vec_float2 = 81619.4

    plt.bar([
        "C in global", 
        "C in shared", 
        "C Bank conflicts",
        "C in registers",
        "Vectorized copy to shared"], 
        [global_c, 
        cache_c,         
        keep_c_bank_confligt,
        keep_c_no_vec, 
        vec_float2])
    plt.ylabel("GFlops")
    plt.title("Optimization strategies for 4096x4096x4096 MMM")
    plt.savefig("optimizations.png", dpi=92, bbox_inches="tight")
    plt.show()

plot_optimizations()
plot_performance()
