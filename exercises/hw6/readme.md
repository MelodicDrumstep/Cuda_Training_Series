## Linked List By Unified Memory

这部分是用 `Unified Memory` 改写一个链表的内存分配。 只需要把


```cpp
template <typename T>
void alloc_bytes(T &ptr, size_t num_bytes)
{
  ptr = (T)malloc(num_bytes);
}
```

改成

```cpp
template <typename T>
void alloc_bytes(T &ptr, size_t num_bytes)
{
  cudaMallocManaged(&ptr, num_bytes);
}
```

然后就可以实现 `CPU / GPU` 兼容:

```c++
  list_elem *list_base, *list;
  alloc_bytes(list_base, sizeof(list_elem));
  list = list_base;
  for (int i = 0; i < num_elem; i++)
  {
    list->key = i;
    alloc_bytes(list->next, sizeof(list_elem));
    list = list->next;
  }
  print_element(list_base, ele); // run on cpu
  gpu_print_element<<<1,1>>>(list_base, ele); // run on gpu
  cudaDeviceSynchronize();
```

## Array Increment Profiling

这个任务的 `kernel` 是这样的：

```c
__global__ void inc(int *array, size_t n)
{
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  while (idx < n)
  {
    array[idx]++;
    idx += blockDim.x * gridDim.x; // grid-stride loop
  }
}
```

这里我用 `nsys` 进行 profiling. 

### nsys output

命令是 `nsys profile --stats=true ./a`

```
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)    Min (ns)   Max (ns)    StdDev (ns)           Name         
 --------  ---------------  ---------  ------------  ------------  --------  -----------  ------------  ---------------------
     61.5      577,184,850         18  32,065,825.0  20,392,438.0     1,365  158,902,064  44,589,691.8  poll                 
     37.5      352,338,064        608     579,503.4      12,057.0       344   25,593,018   2,504,703.8  ioctl                
      0.6        5,454,900          8     681,862.5      39,063.5    30,977    5,186,427   1,820,128.5  sem_timedwait        
      0.2        1,692,806         39      43,405.3       6,473.0     3,914    1,122,569     178,310.9  mmap64               
      0.1          650,410         14      46,457.9       2,977.0       477      406,709     108,924.7  write                
      0.0          395,156          4      98,789.0      99,136.0    91,342      105,542       7,267.3  pthread_create       
      0.0          250,036         65       3,846.7       3,473.0       656        9,332       1,885.6  open64               
      0.0          227,851         29       7,856.9         947.0       521      196,554      36,296.1  fclose               
      0.0          153,509         35       4,386.0       2,256.0     1,239       25,398       5,620.7  fopen                
      0.0           94,532         15       6,302.1       2,486.0       966       44,297      10,823.0  mmap                 
      0.0           40,721         52         783.1          46.0        41       38,218       5,293.1  fgets                
      0.0           40,151         78         514.8         316.0       136       10,881       1,233.2  fcntl                
      0.0           28,941          7       4,134.4       3,852.0       205        9,898       4,061.2  fread                
      0.0           20,292          7       2,898.9       3,284.0     1,412        4,508       1,108.7  open                 
      0.0           19,620          7       2,802.9       2,986.0     1,441        4,420         996.1  munmap               
      0.0           10,259         17         603.5         477.0       252        1,569         388.0  read                 
      0.0           10,165          2       5,082.5       5,082.5     3,903        6,262       1,668.1  socket               
      0.0            9,663          3       3,221.0       2,514.0     2,198        4,951       1,506.5  pipe2                
      0.0            6,302          1       6,302.0       6,302.0     6,302        6,302           0.0  connect              
      0.0            3,510         64          54.8          27.0        25          291          55.0  pthread_mutex_trylock
      0.0            2,529         10         252.9         256.5       167          383          63.9  dup                  
      0.0            1,600          1       1,600.0       1,600.0     1,600        1,600           0.0  bind                 
      0.0              747          1         747.0         747.0       747          747           0.0  listen               

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)        Name      
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  -----------  ----------------
     93.1      378,047,013          1  378,047,013.0  378,047,013.0  378,047,013  378,047,013          0.0  cudaMalloc      
      6.9       27,978,890          2   13,989,445.0   13,989,445.0   13,976,909   14,001,981     17,728.6  cudaMemcpy      
      0.0           28,939          1       28,939.0       28,939.0       28,939       28,939          0.0  cudaLaunchKernel

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)            Name           
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -------------------------
    100.0          423,200          1  423,200.0  423,200.0   423,200   423,200          0.0  inc(int *, unsigned long)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)      Operation     
 --------  ---------------  -----  ------------  ------------  ----------  ----------  -----------  ------------------
     50.9       13,866,983      1  13,866,983.0  13,866,983.0  13,866,983  13,866,983          0.0  [CUDA memcpy HtoD]
     49.1       13,399,464      1  13,399,464.0  13,399,464.0  13,399,464  13,399,464          0.0  [CUDA memcpy DtoH]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)      Operation     
 ----------  -----  --------  --------  --------  --------  -----------  ------------------
    134.218      1   134.218   134.218   134.218   134.218        0.000  [CUDA memcpy DtoH]
    134.218      1   134.218   134.218   134.218   134.218        0.000  [CUDA memcpy HtoD]
```

这里除了显示了 `kernel` 执行性能信息， 还有各个系统调用的执行信息， API 的调用信息， 以及 GPU 内存操作的信息。

还可以执行 `nsys nvprof ./a`:

```
[4/7] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)        Name      
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  -----------  ----------------
     92.7      356,727,165          1  356,727,165.0  356,727,165.0  356,727,165  356,727,165          0.0  cudaMalloc      
      7.3       28,003,092          2   14,001,546.0   14,001,546.0   13,854,383   14,148,709    208,119.9  cudaMemcpy      
      0.0           32,998          1       32,998.0       32,998.0       32,998       32,998          0.0  cudaLaunchKernel

[5/7] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)            Name           
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -------------------------
    100.0          425,888          1  425,888.0  425,888.0   425,888   425,888          0.0  inc(int *, unsigned long)

[6/7] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)      Operation     
 --------  ---------------  -----  ------------  ------------  ----------  ----------  -----------  ------------------
     51.4       14,034,279      1  14,034,279.0  14,034,279.0  14,034,279  14,034,279          0.0  [CUDA memcpy HtoD]
     48.6       13,254,215      1  13,254,215.0  13,254,215.0  13,254,215  13,254,215          0.0  [CUDA memcpy DtoH]

[7/7] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)      Operation     
 ----------  -----  --------  --------  --------  --------  -----------  ------------------
    134.218      1   134.218   134.218   134.218   134.218        0.000  [CUDA memcpy DtoH]
    134.218      1   134.218   134.218   134.218   134.218        0.000  [CUDA memcpy HtoD]
```

这基本上就是 `nsys profile --stats=true` 的子集。

我们摘取关键信息:

```
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)            Name           
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -------------------------
    100.0          425,888          1  425,888.0  425,888.0   425,888   425,888          0.0  inc(int *, unsigned long)
```

### rewrite with unified memory !!

然后我用 `unified memory` 进行重写， 并进行性能测试。

```
用 unified memory 重写的时候别忘了在 host 调用完 kernel 之后 `cudaDeviceSynchronize()`!!

因为 kernel 执行对于 Host 来说是异步的， cudaDeviceSynchronize 是各个设备之间的同步点， 可以确保 kernel 已经执行成功， 从而 host 可以进行测试了。 如果不加这个命令， 则 kernel 还在计算中， host 就开始测试了， 必然会导致测试结果错误。
```

性能分析结果如下:

```
Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)    Min (ns)   Max (ns)    StdDev (ns)           Name         
 --------  ---------------  ---------  ------------  ------------  --------  -----------  ------------  ---------------------
     69.7      832,431,316         35  23,783,751.9  10,296,842.0     1,842  242,615,321  45,277,923.0  poll                 
     23.3      278,556,710        609     457,400.2      12,041.0       416   39,947,690   2,156,302.3  ioctl                
      6.7       80,307,548         31   2,590,566.1   2,128,835.0       102   20,813,289   4,310,521.8  sem_timedwait        
      0.1        1,711,965         39      43,896.5       5,499.0     3,143    1,156,134     183,739.2  mmap64               
      0.0          415,998          4     103,999.5     104,463.0    91,433      115,639       9,963.0  pthread_create       
      0.0          245,671         65       3,779.6       3,386.0       687       10,051       1,871.0  open64               
      0.0          157,123         35       4,489.2       2,727.0     1,198       44,433       7,505.4  fopen                
      0.0          109,916         16       6,869.8       2,834.5     1,089       41,715       9,858.1  mmap                 
      0.0          109,777         29       3,785.4         947.0       535       78,346      14,350.8  fclose               
      0.0           42,246         78         541.6         321.5       133       12,679       1,432.5  fcntl                
      0.0           41,115         52         790.7          47.0        41       38,624       5,349.4  fgets                
      0.0           36,089          7       5,155.6       4,677.0       185       14,835       5,502.7  fread                
      0.0           29,533         14       2,109.5       2,385.5       438        3,275         898.2  write                
      0.0           21,455          7       3,065.0       3,213.0       897        6,246       1,715.6  open                 
      0.0           18,342          5       3,668.4       3,626.0     2,741        4,929         816.1  munmap               
      0.0           11,073         17         651.4         432.0       255        2,395         538.0  read                 
      0.0            9,921          3       3,307.0       3,196.0     1,954        4,771       1,411.8  pipe2                
      0.0            8,280          2       4,140.0       4,140.0     4,051        4,229         125.9  socket               
      0.0            5,500          1       5,500.0       5,500.0     5,500        5,500           0.0  connect              
      0.0            3,082         64          48.2          26.0        25          284          55.4  pthread_mutex_trylock
      0.0            2,188         10         218.8         227.5       138          321          62.1  dup                  
      0.0            1,108          1       1,108.0       1,108.0     1,108        1,108           0.0  bind                 
      0.0              582          1         582.0         582.0       582          582           0.0  listen    

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)           Name         
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  -----------  ----------------------
     93.1      369,166,606          1  369,166,606.0  369,166,606.0  369,166,606  369,166,606          0.0  cudaMallocManaged     
      6.8       26,995,689          1   26,995,689.0   26,995,689.0   26,995,689   26,995,689          0.0  cudaDeviceSynchronize 
      0.1          330,032          1      330,032.0      330,032.0      330,032      330,032          0.0  cudaLaunchKernel      
      0.0            3,575          1        3,575.0        3,575.0        3,575        3,575          0.0  cuModuleGetLoadingMode

[5/7] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)            Name           
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  -------------------------
    100.0       26,993,179          1  26,993,179.0  26,993,179.0  26,993,179  26,993,179          0.0  inc(int *, unsigned long)

[6/7] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)              Operation            
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ---------------------------------
     61.9       10,973,066  2,323   4,723.7   2,880.0     2,335    39,040      5,927.3  [CUDA Unified Memory memcpy HtoD]
     38.1        6,767,391    768   8,811.7   3,280.0     1,823    43,616     11,774.9  [CUDA Unified Memory memcpy DtoH]

[7/7] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)              Operation            
 ----------  -----  --------  --------  --------  --------  -----------  ---------------------------------
    134.218    768     0.175     0.033     0.004     1.044        0.301  [CUDA Unified Memory memcpy DtoH]
    134.218  2,323     0.058     0.008     0.004     0.918        0.150  [CUDA Unified Memory memcpy HtoD]
```

我们把各项指标前后对比一下:

+ kernel execution time

```
// WITHOUT UNIFIED MEMORY
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)            Name           
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -------------------------
    100.0          425,888          1  425,888.0  425,888.0   425,888   425,888          0.0  inc(int *, unsigned long)

// WITH UNIFIED MEMORY
 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)            Name           
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  -------------------------
    100.0       26,993,179          1  26,993,179.0  26,993,179.0  26,993,179  26,993,179          0.0  inc(int *, unsigned long)
```

Ooops! 虽然已经有预期会慢了， 但是这也慢了太多了！ 在数组大小为 `32ULL*1024ULL*1024ULL` 的情况下， 整整慢了 `63` 倍！ 这是不可接受的。

+ memory operation time

```
// WITHOUT UNIFIED MEMORY
 Time (%)  Total Time (ns)  Count    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)      Operation     
 --------  ---------------  -----  ------------  ------------  ----------  ----------  -----------  ------------------
     51.4       14,034,279      1  14,034,279.0  14,034,279.0  14,034,279  14,034,279          0.0  [CUDA memcpy HtoD]
     48.6       13,254,215      1  13,254,215.0  13,254,215.0  13,254,215  13,254,215          0.0  [CUDA memcpy DtoH]

// WITH UNIFIED MEMORY
 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)              Operation            
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ---------------------------------
     61.9       10,973,066  2,323   4,723.7   2,880.0     2,335    39,040      5,927.3  [CUDA Unified Memory memcpy HtoD]
     38.1        6,767,391    768   8,811.7   3,280.0     1,823    43,616     11,774.9  [CUDA Unified Memory memcpy DtoH]
```

这里 `memory operation` 耗时倒没有什么差异。 毕竟这里只统计了 `host` 和 `device` 双向复制这个过程所消耗的时间。 甚至因为 `lazy` 的思想， `unified memory` 版本的 `memory operation` 耗时要少很多。

### system call

```
// WITHOUT UNIFIED MEMORY
Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)    Min (ns)   Max (ns)    StdDev (ns)           Name         
--------  ---------------  ---------  ------------  ------------  --------  -----------  ------------  ---------------------
    61.5      577,184,850         18  32,065,825.0  20,392,438.0     1,365  158,902,064  44,589,691.8  poll                 
    37.5      352,338,064        608     579,503.4      12,057.0       344   25,593,018   2,504,703.8  ioctl    

// WITH UNIFIED MEMORY
Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)    Min (ns)   Max (ns)    StdDev (ns)           Name         
 --------  ---------------  ---------  ------------  ------------  --------  -----------  ------------  ---------------------
     69.7      832,431,316         35  23,783,751.9  10,296,842.0     1,842  242,615,321  45,277,923.0  poll                 
     23.3      278,556,710        609     457,400.2      12,041.0       416   39,947,690   2,156,302.3  ioctl      
```

这里 `poll` 系统调用和 `ioctl` 系统调用是耗时最多的两个。 下面我需要了解一下这两个系统调用。

+ poll

`poll` 是一种多路复用的 IO 模型， 用于监控多个文件描述符， 查看哪些文件描述符可以进行 IO 操作。 它经常用于网络编程中。

+ ioctl

`ioctl` 是设备控制接口， 用于发送控制命令给设备驱动程序。 

因此， 这里性能的下降和这两个系统调用都有关系， 它们都负责了 `host` 与 `device(GPU)` 之间的 IO. 由于初始未分配内存， 因此初次访问会 `page fault`. 而 `GPU` 更新了内存后， `CPU` 访问时又需要从 `GPU` 传输数据， 造成多次 IO 的浪费。

### Prefetching Suggestion!

现在我们考虑加入一些建议命令， 告诉程序我们希望在一开始进行内存拷贝。 我们可以使用 `cudaMemPrefetchAsync(ptr, length, destDevice, stream = default)`

```c
int *h_array;
alloc_bytes(h_array, ds*sizeof(h_array[0]));

cudaCheckErrors("cudaMalloc Error");
memset(h_array, 0, ds*sizeof(h_array[0]));

cudaMemPrefetchAsync(h_array, ds * sizeof(h_array[0]), 0);
// Suggest prefetching the whole array from CPU to GPU
// to avoid page faults

cudaCheckErrors("cudaMemcpy H->D Error");
inc<<<256, 256>>>(h_array, ds);
cudaCheckErrors("kernel launch error");

cudaMemPrefetchAsync(h_array, ds * sizeof(h_array[0]), cudaCpuDeviceId);
// Suggest prefetching the whole array back to CPU
// to avoid page faults

cudaDeviceSynchronize();
// Remember to synchronize
// Otherwise the device is still computing and host starts checking
```

再来看看性能分析:

```
 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)    Min (ns)   Max (ns)    StdDev (ns)           Name         
 --------  ---------------  ---------  ------------  ------------  --------  -----------  ------------  ---------------------
     65.6      815,494,147         33  24,711,943.8  10,319,913.0     1,289  252,765,072  48,628,633.5  poll                 
     25.6      317,691,819        611     519,953.9      12,208.0       341   47,531,842   2,615,690.0  ioctl                
      7.7       95,695,580         27   3,544,280.7     538,557.0        68   20,813,161   5,428,556.7  sem_timedwait        
      0.8       10,419,232          2   5,209,616.0   5,209,616.0    97,424   10,321,808   7,229,731.3  sem_wait             
      0.1        1,741,083         39      44,643.2       5,713.0     3,664    1,174,691     186,647.5  mmap64               
      0.1          810,317          5     162,063.4     109,545.0    96,328      364,494     114,258.9  pthread_create       
      0.0          243,028         65       3,738.9       3,343.0       722        9,259       1,769.7  open64               
      0.0          160,256         35       4,578.7       2,671.0     1,247       45,301       7,689.0  fopen                
      0.0          139,520         29       4,811.0         922.0       541      107,407      19,740.9  fclose               
      0.0          107,409         16       6,713.1       3,128.0       932       44,241      10,391.2  mmap                 
      0.0           41,548         52         799.0          46.0        41       39,076       5,412.2  fgets                
      0.0           40,579         78         520.2         287.0       136       12,002       1,360.2  fcntl                
      0.0           36,822          7       5,260.3       4,804.0       202       14,602       5,537.6  fread                
      0.0           30,289         15       2,019.3       2,104.0       467        3,290         821.0  write                
      0.0           21,110          7       3,015.7       3,243.0       924        5,813       1,543.7  open                 
      0.0           19,757          7       2,822.4       2,480.0     1,225        4,436       1,189.8  munmap               
      0.0           11,499         18         638.8         442.0       236        2,319         521.8  read                 
      0.0            9,851          3       3,283.7       2,717.0     1,578        5,556       2,048.6  pipe2                
      0.0            8,034          2       4,017.0       4,017.0     3,686        4,348         468.1  socket               
      0.0            5,526          1       5,526.0       5,526.0     5,526        5,526           0.0  connect              
      0.0            2,971         64          46.4          26.0        25          304          49.7  pthread_mutex_trylock
      0.0            2,319         10         231.9         248.5       142          296          53.3  dup                  
      0.0            1,077          1       1,077.0       1,077.0     1,077        1,077           0.0  bind                 
      0.0              643          1         643.0         643.0       643          643           0.0  listen               

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)           Name         
 --------  ---------------  ---------  -------------  -------------  -----------  -----------  -----------  ----------------------
     85.5      267,347,469          1  267,347,469.0  267,347,469.0  267,347,469  267,347,469          0.0  cudaMallocManaged     
     12.4       38,887,503          1   38,887,503.0   38,887,503.0   38,887,503   38,887,503          0.0  cudaDeviceSynchronize 
      1.2        3,815,783          2    1,907,891.5    1,907,891.5      253,167    3,562,616  2,340,133.8  cudaMemPrefetchAsync  
      0.8        2,450,692          1    2,450,692.0    2,450,692.0    2,450,692    2,450,692          0.0  cudaLaunchKernel      
      0.0            3,664          1        3,664.0        3,664.0        3,664        3,664          0.0  cuModuleGetLoadingMode

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)            Name           
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -------------------------
    100.0          426,560          1  426,560.0  426,560.0   426,560   426,560          0.0  inc(int *, unsigned long)

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)              Operation            
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ---------------------------------
     52.4        6,028,831     64  94,200.5  93,344.0    92,800    98,400      1,856.9  [CUDA Unified Memory memcpy DtoH]
     47.6        5,485,405     64  85,709.5  85,695.5    85,376    87,103        291.0  [CUDA Unified Memory memcpy HtoD]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)              Operation            
 ----------  -----  --------  --------  --------  --------  -----------  ---------------------------------
    134.218     64     2.097     2.097     2.097     2.097        0.000  [CUDA Unified Memory memcpy DtoH]
    134.218     64     2.097     2.097     2.097     2.097        0.000  [CUDA Unified Memory memcpy HtoD]
```

省流版:

+ kernel execution time

```
// WITHOUT UNIFIED MEMORY
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)            Name           
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -------------------------
    100.0          425,888          1  425,888.0  425,888.0   425,888   425,888          0.0  inc(int *, unsigned long)

// WITH UNIFIED MEMORY
 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)            Name           
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  -------------------------
    100.0       26,993,179          1  26,993,179.0  26,993,179.0  26,993,179  26,993,179          0.0  inc(int *, unsigned long)

// WITH UNIFIED MEMORY ALONG WITH PREFETCHING
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)            Name           
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -------------------------
    100.0          426,560          1  426,560.0  426,560.0   426,560   426,560          0.0  inc(int *, unsigned long)
```

Oh!! 我们使用了 `Prefetching Suggestion` 之后， 性能和最初不使用 `unified memory` 的版本基本一致了！ 这样我们就实现了既有 `unified memory` 的高级抽象和便捷性， 又有最初原始版本的性能。 太棒了！

+ memory operation time

```
// WITHOUT UNIFIED MEMORY
 Time (%)  Total Time (ns)  Count    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)      Operation     
 --------  ---------------  -----  ------------  ------------  ----------  ----------  -----------  ------------------
     51.4       14,034,279      1  14,034,279.0  14,034,279.0  14,034,279  14,034,279          0.0  [CUDA memcpy HtoD]
     48.6       13,254,215      1  13,254,215.0  13,254,215.0  13,254,215  13,254,215          0.0  [CUDA memcpy DtoH]

// WITH UNIFIED MEMORY
 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)              Operation            
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ---------------------------------
     61.9       10,973,066  2,323   4,723.7   2,880.0     2,335    39,040      5,927.3  [CUDA Unified Memory memcpy HtoD]
     38.1        6,767,391    768   8,811.7   3,280.0     1,823    43,616     11,774.9  [CUDA Unified Memory memcpy DtoH]

// WITH UNIFIED MEMORY ALONG WITH PREFETCHING
 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)              Operation            
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ---------------------------------
     52.4        6,028,831     64  94,200.5  93,344.0    92,800    98,400      1,856.9  [CUDA Unified Memory memcpy DtoH]
     47.6        5,485,405     64  85,709.5  85,695.5    85,376    87,103        291.0  [CUDA Unified Memory memcpy HtoD]
```

这里 `memory operation` 用时显著变少， 我猜测是因为驱动有一些潜在的优化?

### system call

```
// WITHOUT UNIFIED MEMORY
Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)    Min (ns)   Max (ns)    StdDev (ns)           Name         
--------  ---------------  ---------  ------------  ------------  --------  -----------  ------------  ---------------------
    61.5      577,184,850         18  32,065,825.0  20,392,438.0     1,365  158,902,064  44,589,691.8  poll                 
    37.5      352,338,064        608     579,503.4      12,057.0       344   25,593,018   2,504,703.8  ioctl    

// WITH UNIFIED MEMORY
Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)    Min (ns)   Max (ns)    StdDev (ns)           Name         
 --------  ---------------  ---------  ------------  ------------  --------  -----------  ------------  ---------------------
     69.7      832,431,316         35  23,783,751.9  10,296,842.0     1,842  242,615,321  45,277,923.0  poll                 
     23.3      278,556,710        609     457,400.2      12,041.0       416   39,947,690   2,156,302.3  ioctl     

// WITH UNIFIED MEMORY ALONG WITH PREFETCHING
 Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)    Min (ns)   Max (ns)    StdDev (ns)           Name         
 --------  ---------------  ---------  ------------  ------------  --------  -----------  ------------  ---------------------
     65.6      815,494,147         33  24,711,943.8  10,319,913.0     1,289  252,765,072  48,628,633.5  poll                 
     25.6      317,691,819        611     519,953.9      12,208.0       341   47,531,842   2,615,690.0  ioctl    
```

这里我们可以发现 `poll` 和 `ioctl` 的用时还是很长， 但是这部分没有算入 `kernel` 执行时间里， 而是在 `prefetching` 阶段做的。 这样避免了阻塞关键路径。

### Exec Kernel Multiple Times

接下来我多次执行 `kernel`:

```c
for(int i = 0; i < 100; i++)
{
  inc<<<256, 256>>>(h_array, ds);
}
```

`kernel` 执行时间是:

```
 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)            Name           
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  -------------------------
    100.0       40,046,257        100  400,462.6  400,368.0   397,153   404,960      1,498.2  inc(int *, unsigned long)
```

我可以发现， 执行 100 次 `kernel` 比执行 1 次的平均用时要少。 这也很好理解: 我每次都操作相同的内存地址， 而只有第一次 `kernel` 调用会触发 `page fault`. 因此第一次的用时肯定会更长一些。

## 总结

通过这个 lab， 学习到了 `unified memory` 的思想， 以及通过建议命令进行更细颗粒度的控制。