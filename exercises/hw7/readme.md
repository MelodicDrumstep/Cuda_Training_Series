# Overlap Cuda Runtime with Kernel Call

## 环境

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 545.23.08              Driver Version: 545.23.08    CUDA Version: 12.3     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A100-SXM4-40GB          On  | 00000000:4B:00.0 Off |                    0 |
| N/A   44C    P0              55W / 400W |      4MiB / 40960MiB |      0%      Default |
|                                         |                      |             Disabled |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

## nsight system GUI

由于我自己的电脑没有英伟达显卡， 因此无法使用 `nsight system` 的 GUI。 因此就按照任务书的指引做一下 `overlap` 优化。

这里的情景可以类比为 $N$ 维向量的按元素加法。 每个元素位都有 3 个操作: `cudaMemcpyHostToDevice`, `kernel launch`, `cudaMemcpyDeviceToHost`. 而这里元素位之间是没有依赖关系的， 因此元素和元素之间操作次序对结果无影响。 所以我可以将操作
划分为多个 `stream`, 每个 `stream` 内部保持操作的次序性， 而不同 `stream` 之间的操作可以任意重叠。

划分 `stream` 分开执行之后， 可以发现 `cuda runtime` 与 `kernel` 可以重叠执行， 而最初的版本是完全没有这种重叠的。 因此， 如果 `stream` 带来的时间延迟可忽略的话， 优化性能是必然的。

代码可以改成这样:

```c
cudaStream_t streams[num_streams];
for (int i = 0; i < num_streams; i++) 
{
    cudaStreamCreate(&streams[i]);
}

//...

for (int i = 0; i < chunks; i++) 
{ //depth-first launch
    cudaMemcpyAsync(d_x + i * (ds / chunks), h_x + i * (ds / chunks), (ds / chunks) * sizeof(ft), cudaMemcpyHostToDevice, streams[i * num_streams / chunks]);
    gaussian_pdf<<<((ds / chunks) + 255) / 256, 256, 0, streams[i * num_streams / chunks]>>>(d_x + i * (ds / chunks), d_y + i * (ds / chunks), 0.0, 1.0, ds / chunks);
    cudaMemcpyAsync(h_y + i * (ds / chunks), d_y + i * (ds / chunks), (ds / chunks) * sizeof(ft), cudaMemcpyDeviceToHost, streams[i * num_streams / chunks]);
}
```


图示：

<img src="https://notes.sjtu.edu.cn/uploads/upload_08dcb7cac650e2c9b66f7c9f762f53ef.png" width="500">


编译命令:

```
nvcc -o overlap overlap.cu -DUSE_STREAMS
```

运行结果

```
non-stream elapsed time: 0.025666
streams elapsed time: 0.014049
```

可以看到大致快了接近一倍， 符合预期。