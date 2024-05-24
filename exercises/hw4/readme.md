这个 lab 是写 `matrix_row_sum` / `matrix_col_sum`, 并做 profiling.

首先， 代码非常容易:

```c
// matrix row-sum kernel
__global__ void row_sums(const float *A, float *sums, size_t ds)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x; // create typical 1D thread index from built-in variables
  if (idx < ds)
  {
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++)
    {
      sum += A[idx * ds + i];  
    }       // write a for loop that will cause the thread to iterate across a row, keeeping a running sum, and write the result to sums
    sums[idx] = sum;
  }
}

// matrix column-sum kernel
__global__ void column_sums(const float *A, float *sums, size_t ds)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x; // create typical 1D thread index from built-in variables
  if (idx < ds)
  {
    float sum = 0.0f;
    for (size_t i = 0; i < ds; i++)
    {
      sum += A[i * ds + idx];
    }         // write a for loop that will cause the thread to iterate down a column, keeeping a running sum, and write the result to sums
    sums[idx] = sum;
  }
}
```

接下来是 profiling 部分。 首先我这里的猜测是 `row_sum` 更快， 毕竟缓存命中率高嘛。先用 `nv-nsight-cu-cli`:

来看看结果:

```
 row_sums(const float *, float *, unsigned long) (64, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         1.21
    SM Frequency            cycle/nsecond         1.09
    Elapsed Cycles                  cycle    4,205,684
    Memory Throughput                   %        59.11
    DRAM Throughput                     %        17.98
    Duration                      msecond         3.84
    L1/TEX Cache Throughput             %        99.84
    L2 Cache Throughput                 %        17.06
    SM Active Cycles                cycle 2,489,585.13
    Compute (SM) Throughput             %         2.08
    ----------------------- ------------- ------------

  column_sums(const float *, float *, unsigned long) (64, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         1.21
    SM Frequency            cycle/nsecond         1.09
    Elapsed Cycles                  cycle    2,957,480
    Memory Throughput                   %        25.56
    DRAM Throughput                     %        25.56
    Duration                      msecond         2.71
    L1/TEX Cache Throughput             %         9.17
    L2 Cache Throughput                 %        21.48
    SM Active Cycles                cycle 1,694,262.75
    Compute (SM) Throughput             %         3.94
    ----------------------- ------------- ------------
```

Oh 看来我猜错了！实际上是 `col_sum` 更快的， 快了接近 1.5 倍。 为什么会这样？ 实际上这个结合本次课所讲的 `memory coalescing` 就好理解了。 为什么我直接猜测是 `row_sum` 更快？ 实际上那是单线程的经验。 在单线程的情况下， 按行遍历缓存命中率更高。 但是 cuda 是多线程模型， 这里如果采取 `row_sum`， 则同一时刻同一 `warp` 的不同 `thread` 访问的内存地址的 __连续__ 的。 这样 memory controller 将内存访问打包为 `transaction` 后， 访存次数就少了很多。 但如果采取 `row_sum` 这里同一时刻同一 `warp` 的不同 `thread` 访问的内存地址是分散的， 需要非常多个 `transaction` 才能访问完这些内存， 效率会降低。

为什么这里没分析到缓存命中率？ 因为虽然 `row_sum` 每个线程第一次访问内存之后都将这一行的多个元素以 `cacheline` 的形式移入了 `cache`， 但是之后访问仍要多次访问 `cache`， 每一次都不能合并访问， 还是浪费了很多次访存次数。 而 `col_sum` 每一次访存都是合并访问， 极大地减少了访存次数。 具体计算而言， 假设 `cacheline` 大小为 `L`, 矩阵为 `N x N`, 则有 `N` 个线程。 那么 `col_sum` 经过 `memory coalescing` 之后只有 `N` 次访存， 而 `row_sum` 有 `N * (N / L)` 次访存和 `N * (N - N / L)` 次访问 `cache`.

还可以用这个命令来具体分析访存:

```
nv-nsight-cu-cli --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./matrix_sums
```

结果如下:

```
  row_sums(const float *, float *, unsigned long) (64, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.0
    Section: Command line profiler metrics
    ----------------------------------------------- ----------- ------------
    Metric Name                                     Metric Unit Metric Value
    ----------------------------------------------- ----------- ------------
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum     request    8,388,608
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       sector  268,435,456
    ----------------------------------------------- ----------- ------------

  column_sums(const float *, float *, unsigned long) (64, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.0
    Section: Command line profiler metrics
    ----------------------------------------------- ----------- ------------
    Metric Name                                     Metric Unit Metric Value
    ----------------------------------------------- ----------- ------------
    l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum     request    8,388,608
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum       sector   33,554,432
    ----------------------------------------------- ----------- ------------
```

此处， `request` 表示请求的内存访问数量， `sector` 表示经过 memory controller 之后的 `transaction`数量。这里由于都是遍历一个二维数组， 因此自然 `request` 是一样的。 而 `col_sum` 有 `memory coalescning`， 因此最终的 `transaction` 数量远小于 `row_sum`， 性能也快了非常多。(16000 * 16000 的矩阵下快了 1.5倍)


