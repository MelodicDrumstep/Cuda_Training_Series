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
