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


