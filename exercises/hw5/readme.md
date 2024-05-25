这个 lab 是学习 `reduction`, `atomic operation`, `warp shuffle`。

课上讲到了这个 "并行求向量所有元素之和" 的程序, 我为它写了逐行注释:


```cpp
#include <stdio.h>

const size_t N = 8ULL * 1024ULL * 1024ULL;
const int BLOCK_SIZE = 256;

__global__ void reduce_sum(float * d_A, float * d_sum, size_t N)
{
    __shared__ float sdata[BLOCK_SIZE];
    //Create a shared memory of BLOCK_SIZE
    //Each thread in this block
    //will fill in a single value in this shared memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    for(int i = idx; i < N; i += blockDim.x * gridDim.x)
    {
        sdata[tid] += d_A[i];
        //This is a grid-stride loop
        //This will add up elements by a stride of blockDim.x * gridDim.x
        //then store it to the shared memory
        //This step reduce the array size from
        //N to blockDim.x * gridDim.x
    }

    //Following is a reduction step:
    //Each time we reduce the array size by half
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        __syncthreads();
        //Call __syncthreads first
        //At the first time we can sync from the grid-stride loop
        //And at each loop we can sync from every threads in the block
        //This is to make sure that all threads have finished the previous step
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
            //For the first half of the threads
            //We add the second half of the threads
            //This type of memory accesing can be coalesced
        }
    }
    //Af the loop, each block finish its reduction
    //Then for the shared memory of each block
    //the first element store the sum of that block
    if(tid == 0)
    {
        d_sum[blockIdx.x] = sdata[0];
        //Then we output the sum of each block
        //The whole output array is of size "blocks", i.e. "gridDim.x"
    }
}


int main()
{
    float * h_A, * h_sum, * d_A, * d_sums;
    const int blocks = 640;
    //This is the number of blocks
    //i.e. gridDim.x
    h_A = new float[N];
    h_sum = new float;
    for(size_t i = 0; i < N; i++)
    {
        h_A[i] = 1.0f;
    }
    float maxval = 5.0f;
    h_A[N / 2] = maxval;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_sums, blocks * sizeof(float));
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    reduce_sum<<<blocks, BLOCK_SIZE>>>(d_A, d_sums, N);
    //At the first kernel, we output a partial sum for each block
    //the output array size is "blocks"
    reduce_sum<<<1, BLOCK_SIZE>>>(d_sums, d_A, blocks);
    //Then we call the kernel again to sum up the partial sums
    //The output is stored in d_A[0]
    cudaMemcpy(h_sum, d_A, sizeof(float), cudaMemcpyDeviceToHost);
    printf("reduction output: %f, expected sum reduction output: %f, expected max reduction output: %f\n", * h_sum, (float)((N - 1) + maxval), maxval);
    return 0;
}
```

这个程序第一次 reduction 将大小为 `N` 的数组缩小为大小为 `blocksize` 的数组， 第二次 reduction 将大小为 `blocksize` 的数组缩小为 1 个元素。 最终求出了所有元素之和。 下图对于理解本算法很有帮助:

<img src="https://notes.sjtu.edu.cn/uploads/upload_db124c34148745f93d48f0c41b87b436.png" width="300">

本次 lab 的目的是探究对于 "求向量个元素之和" 的不同实现之间的性能差异。 上述是一种已经较为聪明的实现了。 下面我列举本次 lab 提到的所有实现:

## Naive atomic

```cpp
__global__ void atomic_red(const float *gdata, float *out)
{
  size_t idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx < N) 
  {
      atomicAdd(out, gdata[idx]);
      //Just use atomic add to sum all the elements in the array
      //This is not efficient, but it is correct 
      //And nead a lot of threads
  }
}
```

这种实现很暴力，也很低效。

## simple reduce

即最上方提到的算法

```cpp
__global__ void reduce_sum(float * d_A, float * d_sum, size_t N)
{
    __shared__ float sdata[BLOCK_SIZE];
    //Create a shared memory of BLOCK_SIZE
    //Each thread in this block
    //will fill in a single value in this shared memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    sdata[tid] = 0.0f;
    for(int i = idx; i < N; i += blockDim.x * gridDim.x)
    {
        sdata[tid] += d_A[i];
        //This is a grid-stride loop
        //This will add up elements by a stride of blockDim.x * gridDim.x
        //then store it to the shared memory
        //This step reduce the array size from
        //N to blockDim.x * gridDim.x
    }

    //Following is a reduction step:
    //Each time we reduce the array size by half
    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        __syncthreads();
        //Call __syncthreads first
        //At the first time we can sync from the grid-stride loop
        //And at each loop we can sync from every threads in the block
        //This is to make sure that all threads have finished the previous step
        if(tid < s)
        {
            sdata[tid] += sdata[tid + s];
            //For the first half of the threads
            //We add the second half of the threads
            //This type of memory accesing can be coalesced
        }
    }
    //Af the loop, each block finish its reduction
    //Then for the shared memory of each block
    //the first element store the sum of that block
    if(tid == 0)
    {
        d_sum[blockIdx.x] = sdata[0];
        //Then we output the sum of each block
        //The whole output array is of size "blocks", i.e. "gridDim.x"
    }
}
```

这种实现很好地利用了线程的计算能力， 同时利用 `memory coalescing` 实现了访存加速。 但是这种实现需要两次调用 kernel， 有一定额外开销。

## reduce with atomic

```cpp
 __global__ void reduce_a(float *gdata, float *out)
 {
     __shared__ float sdata[BLOCK_SIZE];
     int tid = threadIdx.x;
     sdata[tid] = 0.0f;
     int idx = threadIdx.x + blockDim.x * blockIdx.x;

     while (idx < N) 
     {  // grid stride loop to load data
        sdata[tid] += gdata[idx];
        idx += gridDim.x * blockDim.x;  
      }

     for (unsigned int s=blockDim.x / 2; s > 0; s >>= 1) 
     {
        __syncthreads();
        if (tid < s)  // parallel sweep reduction
        {
            sdata[tid] += sdata[tid + s];
        }
      }
     if (tid == 0) 
     {
      atomicAdd(out, sdata[0]);
     }
  }
```

这种实现就是把第二种实现的最后几行改成了 `atomicAdd`, 直接在第一个 kernel 内计算出结果， 避免了第二个 kernel 的调用。 但最后的 `atomicAdd` 也会导致强制次序化，可能带来性能损耗。

## warp shuffle

上述实现中， 如果我们需要在同一个 `block` 的不同 `thread` 之间传值， 则需要用到 `shared memory`。 `shared memory` 已经很快了，但还是有不可避免的开销。 我能不能直接把值从一个 `thread` 扔到另一个 `thread`, 而不是先存起来再由另一个 `thread` 访问呢？

如下图， cuda 为我们提供了 `warp shuffle` 机制实现这一功能。这是一个寄存器到寄存器的通信。

<img src="https://notes.sjtu.edu.cn/uploads/upload_43be7954631f5b27249d4c69a327a234.png" width="300">


下面是这种实现的代码，我写了详细注释:

```cpp
__global__ void reduce_ws(float *gdata, float *out)
{
     __shared__ float sdata[32];
     //This 32 is computed by BLOCKSIZE / WARPSIZE
     //namely how many warps are in a single block
     int tid = threadIdx.x;
     int idx = threadIdx.x + blockDim.x*blockIdx.x;
     float val = 0.0f;
     unsigned mask = 0xFFFFFFFFU;
     //This mask means all threads in a warp
     //are participating
     //Then all of them must be sync for warp shuffle
     int lane = threadIdx.x % warpSize;
     //lane means the id of this thread within a warp
     int warpID = threadIdx.x / warpSize;
     //warpID means which warp it's in
     while (idx < N) 
     {  // grid stride loop to load 
        val += gdata[idx];
        idx += gridDim.x * blockDim.x;  
        //Notice that we use a local vairable to 
        //store the sum of the grid-stride loop
        //rather than store it in shared memory
     }

    // 1st warp-shuffle reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) 
    {
       val += __shfl_down_sync(mask, val, offset);
       //shuffle the local variable "val" from 
       //the right "offset" thread to this thread
    }
    //Now the local variable "val" of the id 0 thread in each warp
    //contains the sum of these threads in this warp
    if (lane == 0) 
    {
       sdata[warpID] = val;
    }
    __syncthreads(); //put warp results in shared mem
    //Notice that __syncthread is a block-level sync

    // hereafter, just use all the threads in warp 0 in each block
    //This is because we BLOCKSIZE / WARPSIZE = 32 = WARPSIZE here
    if (warpID == 0)
    {
       // reload val from shared mem if warp existed
       val = (tid < blockDim.x / warpSize) ? sdata[lane] : 0;
       //Here we want to use all the threads within the first warp
       //to continue computing
       //that's because warp shuffle can only be used
       //within a warp

       // final warp-shuffle reduction
       for(int offset = warpSize / 2; offset > 0; offset >>= 1) 
       {
          val += __shfl_down_sync(mask, val, offset);
       }
       if(tid == 0) 
       {
        //every block will issue an atomic add
         atomicAdd(out, val);
       }
   }
}
```

## profiling

来用 `nsight` profiler 对比一下各种实现的性能:

```
ATOMIC_READ:

  atomic_red(const float *, float *) (32768, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- -------------
    Metric Name               Metric Unit  Metric Value
    ----------------------- ------------- -------------
    DRAM Frequency          cycle/nsecond          1.21
    SM Frequency            cycle/nsecond          1.09
    Elapsed Cycles                  cycle    26,247,696
    Memory Throughput                   %          1.64
    DRAM Throughput                     %          0.09
    Duration                      msecond         23.97
    L1/TEX Cache Throughput             %          0.32
    L2 Cache Throughput                 %          1.64
    SM Active Cycles                cycle 26,077,682.10
    Compute (SM) Throughput             %          0.30
    ----------------------- ------------- -------------

REDUCE_A:

 reduce_a(float *, float *) (640, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         1.14
    SM Frequency            cycle/nsecond         1.02
    Elapsed Cycles                  cycle       45,780
    Memory Throughput                   %        51.71
    DRAM Throughput                     %        51.71
    Duration                      usecond        44.67
    L1/TEX Cache Throughput             %        11.95
    L2 Cache Throughput                 %        37.87
    SM Active Cycles                cycle    40,622.57
    Compute (SM) Throughput             %        13.46
    ----------------------- ------------- ------------

REDUCE_WS:

 reduce_ws(float *, float *) (640, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 8.0
    Section: GPU Speed Of Light Throughput
    ----------------------- ------------- ------------
    Metric Name               Metric Unit Metric Value
    ----------------------- ------------- ------------
    DRAM Frequency          cycle/nsecond         1.16
    SM Frequency            cycle/nsecond         1.05
    Elapsed Cycles                  cycle       46,119
    Memory Throughput                   %        51.32
    DRAM Throughput                     %        51.32
    Duration                      usecond        43.97
    L1/TEX Cache Throughput             %        11.94
    L2 Cache Throughput                 %        37.62
    SM Active Cycles                cycle    40,662.38
    Compute (SM) Throughput             %        14.36
    ----------------------- ------------- ------------
```

总结列表如下:

```
kernel              duration           memory throughput
atomic_read          23.97ms                1.64%
reduce_a             44.67us                51.71%
reduce_ws            43.97us                51.32%
```

我们可以清晰地看出， `reduce_a` 和 `reduce_ws` 性能较为相近，都比 `atomic_read` 有显著提升。 在 `const size_t N = 8ULL*1024ULL*1024ULL; ` 下， 性能提升了 `536` 倍左右。

我们现在比较一下 `N / duration` 和 `memory bandwidth`. 如何得到 `memory bandwidth`? 我可以写个脚本跑一下:

```c
#include <iostream>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);

    // Memory bandwith: Memory Clock * Memory Bus Width / 8
    float memoryClock = prop.memoryClockRate * 1e-3; // convert to MHz
    float memoryBusWidth = prop.memoryBusWidth;
    float memoryBandwidth = 2.0 * memoryClock * (memoryBusWidth / 8) / 1e3; // convert to GB/s

    std::cout << "GPU Name: " << prop.name << std::endl;
    std::cout << "Memory Clock Rate (MHz): " << memoryClock << std::endl;
    std::cout << "Memory Bus Width (bits): " << memoryBusWidth << std::endl;
    std::cout << "Memory Bandwidth (GB/s): " << memoryBandwidth << std::endl;

    return 0;
}
```

这样就可以输出 `memory bandwidth` 的具体数值了。 

```
Memory Clock Rate (MHz): 1215
Memory Bus Width (bits): 5120
Memory Bandwidth (GB/s): 1555.2
```

然后计算一下 `N / duration:`

`N / duration = 8 * 1024 * 1024 * 4B / 44.67us = 751162 b/us = 700GB/s`

所以大概是 `memory bandwidth` 的一半，`nsight` 显示的 `memory throughput` 也大概是这个数， 性能已经很不错了。

