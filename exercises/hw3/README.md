这个 lab 是写 grid-stride loop 的 vecadd 并学习如何 profiling.

## grid-stride loop

大致就是这样

```C++
  for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < ds; idx += gridDim.x * blockDim.x)         // a grid-stride loop
  {
    C[idx] = A[idx] + B[idx];
  }       
```

它可以自适应地完成向量加法。

假如我们调用

```cpp
vadd<<<2, 3>>>(d_A, d_B, d_C, DSIZE);
```

那么 `gridDIm.x = 2`, `blockDim.x = 3`, 对于 `threadIdx.x = 0, 1, .., 5` 每个人负责计算 `DSIZE / 6` 部分元素。

## Profiling

这里主要是学习如何使用 `GPU profiler Nsight` 进行 profiling. 

### nsight

指令：`nv-nsight-cu-cli ./executable`

因为是第一次用 `nv` 家的 profiler, 所以这里粘贴一下完整输出：

```
==PROF== Connected to process 2950850 (/dssg/home/acct-hpc/asc/tuwenliang/Cuda/exercises/hw3/vector_add)
==PROF== Profiling "vadd" - 0: 0%....50%....100% - 10 passes
A[0] = 0.840188
B[0] = 0.394383
C[0] = 1.234571
==PROF== Disconnected from process 2950850
[2950850] vector_add@127.0.0.1
  vadd(const float *, const float *, float *, int) (1, 1, 1)x(1, 1, 1), Context 1, Stream 7, Device 0, CC 8.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         1.21
    SM Frequency            cycle/nsecond         1.09
    Elapsed Cycles                  cycle    4,247,581
    Memory Throughput                   %         0.03
    DRAM Throughput                     %         0.00
    Duration                      msecond         3.88
    L1/TEX Cache Throughput             %         2.37
    L2 Cache Throughput                 %         0.03
    SM Active Cycles                cycle    39,311.81
    Compute (SM) Throughput             %         0.02
    ----------------------- ------------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.0 full      
          waves across all SMs. Look at Launch Statistics for more details.                                             

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                     1
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                      1
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    Threads                                   thread               1
    Waves Per SM                                                0.00
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 96.88%                                                                                          
          Threads are executed in groups of 32 threads called warps. This kernel launch is configured to execute 1      
          threads per block. Consequently, some threads in a warp are masked off and those hardware resources are       
          unused. Try changing the number of threads per block to be a multiple of 32 threads. Between 128 and 256      
          threads per block is a good initial range for experimentation. Use smaller thread blocks rather than one      
          large thread block per multiprocessor if latency affects performance.  This is particularly beneficial to     
          kernels that frequently call __syncthreads(). See the Hardware Model                                          
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for more      
          details on launch configurations.                                                                             
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 99.07%                                                                                          
          The grid for this launch is configured to execute only 1 blocks, which is less than the GPU's 108             
          multiprocessors. This can underutilize some multiprocessors. If you do not intend to execute this kernel      
          concurrently with other workloads, consider reducing the block size to have at least one block per            
          multiprocessor or increase the size of the grid to fully utilize the available hardware resources. See the    
          Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model)            
          description for more details on launch configurations.                                                        

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           32
    Block Limit Registers                 block          128
    Block Limit Shared Mem                block           32
    Block Limit Warps                     block           64
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %           50
    Achieved Occupancy                        %         1.56
    Achieved Active Warps Per SM           warp         1.00
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 96.88%                                                                                    
          The difference between calculated theoretical (50.0%) and measured achieved occupancy (1.6%) can be the       
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Local Speedup: 50%                                                                                       
          The 8.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 16. This kernel's theoretical occupancy (50.0%) is limited by the number of blocks that   
          can fit on the SM. This kernel's theoretical occupancy (50.0%) is limited by the required amount of shared    
          memory.                                                                                                   
```

来精细解读一下 profiler 的信息， 从中也可以学到性能优化需要关心的地方:

### Profiler 信息解读

```
[2950850] vector_add@127.0.0.1
  vadd(const float *, const float *, float *, int) (1, 1, 1)x(1, 1, 1), Context 1, Stream 7, Device 0, CC 8.0
```

这里首先给出了函数原型，`(1, 1, 1) x (1, 1, 1)` 给出了 `grid / block` 配置， 即只有 1 个 grid 和 1 个 block. `Device` 是设备编号， `CC （Compute Capability）` 指的是计算能力。

#### Section: GPU Speed Of Light Throughput

```
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         1.21
    SM Frequency            cycle/nsecond         1.09
    Elapsed Cycles                  cycle    4,247,581
    Memory Throughput                   %         0.03
    DRAM Throughput                     %         0.00
    Duration                      msecond         3.88
    L1/TEX Cache Throughput             %         2.37
    L2 Cache Throughput                 %         0.03
    SM Active Cycles                cycle    39,311.81
    Compute (SM) Throughput             %         0.02
    ----------------------- ------------- ------------
```


这里给出了 GPU 的关键性能指标：

```
DRAM 频率：1.21 cycles/nsecond
SM 频率：1.09 cycles/nsecond
总周期数：4,247,581
内存吞吐量：0.03%
DRAM 吞吐量：0.00%
执行时间：3.88 毫秒
L1/TEX 缓存吞吐量：2.37%
L2 缓存吞吐量：0.03%
SM 活跃周期数：39,311.81
计算吞吐量：0.02%
```

然后给出了一个优化提示:

```
OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.0 full      
          waves across all SMs. Look at Launch Statistics for more details.                                           
```

看来是检测到 grid 配置太小了。

#### Section: Launch Statistics

```
    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                     1
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                      1
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block       Kbyte/block            1.02
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    Threads                                   thread               1
    Waves Per SM                                                0.00
    -------------------------------- --------------- ---------------
```

这部分是内核启动配置的详细信息:

```
block 大小：1
函数缓存配置：无特殊偏好
grid 大小：1
每线程寄存器数：16
共享内存配置大小：32.77 KB
每块驱动共享内存：1.02 KB
每块动态共享内存：0 bytes
每块静态共享内存：0 bytes
线程数：1
每个 SM 的波次：0.00
```

然后又给了个优化提示:

```
 OPT   Est. Speedup: 96.88%                                                                                          
          Threads are executed in groups of 32 threads called warps. This kernel launch is configured to execute 1      
          threads per block. Consequently, some threads in a warp are masked off and those hardware resources are       
          unused. Try changing the number of threads per block to be a multiple of 32 threads. Between 128 and 256      
          threads per block is a good initial range for experimentation. Use smaller thread blocks rather than one      
          large thread block per multiprocessor if latency affects performance.  This is particularly beneficial to     
          kernels that frequently call __syncthreads(). See the Hardware Model                                          
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for more      
          details on launch configurations.    
```

这个建议提示了我， GPU 执行指令的基本单位是 warp（此处为 32 个线程）， 因此要把 thread per block 设置为 32 的倍数， 建议设置为 128 / 256.

```
    OPT   Est. Speedup: 99.07%                                                                                          
          The grid for this launch is configured to execute only 1 blocks, which is less than the GPU's 108             
          multiprocessors. This can underutilize some multiprocessors. If you do not intend to execute this kernel      
          concurrently with other workloads, consider reducing the block size to have at least one block per            
          multiprocessor or increase the size of the grid to fully utilize the available hardware resources. See the    
          Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model)            
          description for more details on launch configurations.     
```

这个建议是增加 grid 数量， 以充分利用 GPU 资源。 这个 GPU 有 108 个 multiprocessor.

#### Section: Occupancy

```
  Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           32
    Block Limit Registers                 block          128
    Block Limit Shared Mem                block           32
    Block Limit Warps                     block           64
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %           50
    Achieved Occupancy                        %         1.56
    Achieved Active Warps Per SM           warp         1.00
    ------------------------------- ----------- ------------
```

这部分显示了内核的占用情况：

```
每个 SM 的块限制：32
每个 SM 的寄存器块限制：128
每个 SM 的共享内存块限制：32
每个 SM 的 warp 限制：64
理论上的每个 SM 活跃 warp：32
理论占用率：50%
实际达到的占用率：1.56%
实际活跃的 warp per SM：1.00
```

接着又是一些优化提示:

```
   OPT   Est. Local Speedup: 96.88%                                                                                    
          The difference between calculated theoretical (50.0%) and measured achieved occupancy (1.6%) can be the       
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.        
```

这个提示告诉我， 实际占用率和理论占用率之间的差距可能来源于 warp schedule 的开销， 或者负载的不均衡。

```
 OPT   Est. Local Speedup: 50%                                                                                       
          The 8.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 16. This kernel's theoretical occupancy (50.0%) is limited by the number of blocks that   
          can fit on the SM. This kernel's theoretical occupancy (50.0%) is limited by the required amount of shared    
          memory.
```

这个提示告诉我， 理论占用率受限于每个 SM 上 Block 的数量和共享内存的数量。


### 本次任务 profiling

本次主要关注 `Duration` 和 `Memory Throughput`. 

这里先将参数调大。 此时参数大小为 `const int DSIZE = 32*1048424;`

此时 profile 信息是这样的

#### `grid, block : (1, 1)`

```
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- -------------
    Metric Name               Metric Unit  Metric Value
    ----------------------- ------------- -------------
    DRAM Frequency          cycle/nsecond          1.21
    SM Frequency            cycle/nsecond          1.09
    Elapsed Cycles                  cycle 4,131,941,105
    Memory Throughput                   %          0.03
    DRAM Throughput                     %          0.01
    Duration                       second          3.77
    L1/TEX Cache Throughput             %          2.44
    L2 Cache Throughput                 %          0.03
    SM Active Cycles                cycle 38,265,118.79
    Compute (SM) Throughput             %          0.02
    ----------------------- ------------- -------------
```

现在我去改 block size 和 grid size, 再来看看有什么不同。

#### `grid, block : (1, 1024)`

```
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         1.21
    SM Frequency            cycle/nsecond         1.09
    Elapsed Cycles                  cycle   23,692,869
    Memory Throughput                   %         1.17
    DRAM Throughput                     %         1.17
    Duration                      msecond        21.67
    L1/TEX Cache Throughput             %        26.48
    L2 Cache Throughput                 %         1.03
    SM Active Cycles                cycle   219,430.76
    Compute (SM) Throughput             %         0.12
    ----------------------- ------------- ------------
```

比原先的快了 170 多倍！ 这是只利用了一个 SM 的情况， 下面来看看充分利用所有资源是怎样的效果。


#### `grid, block : (64, 64)`

```
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         1.21
    SM Frequency            cycle/nsecond         1.09
    Elapsed Cycles                  cycle    5,256,493
    Memory Throughput                   %         5.27
    DRAM Throughput                     %         5.27
    Duration                      msecond         4.81
    L1/TEX Cache Throughput             %         1.92
    L2 Cache Throughput                 %         4.63
    SM Active Cycles                cycle 3,025,467.41
    Compute (SM) Throughput             %         0.55
    ----------------------- ------------- ------------
```

比原先的整整快了接近 800 倍！但是 profiler 也提示 grid size, block size 都太小了， 下面改大一点。


#### `grid, block : (160, 1024)`

```
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         1.22
    SM Frequency            cycle/nsecond         1.10
    Elapsed Cycles                  cycle      351,986
    Memory Throughput                   %        78.49
    DRAM Throughput                     %        78.49
    Duration                      usecond       319.97
    L1/TEX Cache Throughput             %        17.16
    L2 Cache Throughput                 %        63.43
    SM Active Cycles                cycle   326,492.26
    Compute (SM) Throughput             %         8.32
    ----------------------- ------------- ------------
```

比最初快了 11782 倍！！ 此时 profiler 提示

```
    OPT   Memory is more heavily utilized than Compute: Look at the Memory Workload Analysis section to identify the    
          DRAM bottleneck. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the       
          bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or  
          whether there are values you can (re)compute.  
```

提示了我访存是一个瓶颈。 我发现这和算法实现有关， 这里类似 `#pragma openmp for` 的实现方式， 是每个线程跳步计算元素。 如果我改成每个线程分块计算元素， 那就增大了缓存命中率。 因此我打算试一试改写这个算法。

先来再试试其他配置:

#### `grid, block : (80, 1024)`

```
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         1.21
    SM Frequency            cycle/nsecond         1.09
    Elapsed Cycles                  cycle      435,250
    Memory Throughput                   %        63.57
    DRAM Throughput                     %        63.57
    Duration                      usecond       399.65
    L1/TEX Cache Throughput             %        17.81
    L2 Cache Throughput                 %        52.42
    SM Active Cycles                cycle   314,852.46
    Compute (SM) Throughput             %         6.71
    ----------------------- ------------- ------------
```

#### `grid, block : (240, 1024)`

```
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         1.21
    SM Frequency            cycle/nsecond         1.09
    Elapsed Cycles                  cycle      400,236
    Memory Throughput                   %        69.17
    DRAM Throughput                     %        69.17
    Duration                      usecond       368.06
    L1/TEX Cache Throughput             %        17.95
    L2 Cache Throughput                 %        55.82
    SM Active Cycles                cycle   310,292.80
    Compute (SM) Throughput             %         7.33
    ----------------------- ------------- ------------
```

看来 `(160, 1024)` 的配置确实是一个近似最优的配置了。 下面来改写算法:

## 并行算法改进

我把我的算法改成了这样

```c
__global__ void vadd(const float *A, const float *B, float *C, int ds, int compute_block_size)
{
    int start = (threadIdx.x + blockDim.x * blockIdx.x) * compute_block_size;
    for(int idx = start; idx < ds && idx < start + compute_block_size; idx++)
    {
        C[idx] = A[idx] + B[idx];
    } 
}
```

然后在 `(160, 1024)` 的配置下跑了一下:

```
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         1.21
    SM Frequency            cycle/nsecond         1.09
    Elapsed Cycles                  cycle    6,374,606
    Memory Throughput                   %        42.64
    DRAM Throughput                     %        35.76
    Duration                      msecond         5.86
    L1/TEX Cache Throughput             %        21.18
    L2 Cache Throughput                 %        54.11
    SM Active Cycles                cycle 5,276,230.03
    Compute (SM) Throughput             %         1.57
    ----------------------- ------------- ------------
```

为什么会比第一个方案慢这么多？？

我去查阅了一些资料， 这和 `memory coalescing` 有关。 当我们选用 `grid-stride loop` 时， 我们可以保证一个 warp 内的内存访问都是 `unit-stride`， 因此便于 `memory coalescing`. 

推荐阅读： 

1.https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/

2.https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/

总结下来， `grid-stride loop` 有哪些好处呢？

+ scalability and thread reuse.

这种写法可以拓展到任何大小的问题， 甚至哪怕问题规模大于了 CUDA 设备的支持规模。 同时， 调参也更加方便。 

当我们限制住总线程数的时候， thread 可以被重复利用来计算多个元素， 这便是 thread reuse.

thread reuse 又有什么好处？ 我们知道， 创建线程和销毁线程是有开销的， 而 thread reuse 就可以分摊这份开销。

+ 利于 debug

只要我把 grid size 和 block size 都设置为 1， 那么就是顺序执行的情况， 非常方便 debug.

+ 可移植性和易读性

它和原始的顺序执行的算法很类似，非常容易理解。 同时可移植性非常好。

我们甚至可以利用 `grid-stride loop` 写一份 cuda 代码， 它可以同时在 CPU 和 GPU 上运行。

我们使用 `Hemi` 库:

```c
HEMI_LAUNCHABLE
void saxpy(int n, float a, float *x, float *y)
{
  for (auto i : hemi::grid_stride_range(0, n)) {
    y[i] = a * x[i] + y[i];
  }
}
```

如果在 CPU 上运行， 上述就是一个 function call, 如果在 GPU 上运行， 上述就是一个 kernerl call.

```c
hemi::cudaLaunch(saxpy, 1<<20, 2.0, x, y);
```



