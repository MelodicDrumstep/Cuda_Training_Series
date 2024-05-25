#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const size_t N = 8ULL*1024ULL*1024ULL;  // data size
//const size_t N = 256*640; // data size
const int BLOCK_SIZE = 256;  // CUDA maximum is 1024
// naive atomic reduction kernel
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


__global__ void reduce(float * d_A, float * d_sum, size_t N)
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




int main(){

  float *h_A, *h_sum, *d_A, *d_sum;
  h_A = new float[N];  // allocate space for data in host memory
  h_sum = new float;
  for (int i = 0; i < N; i++)  // initialize matrix in host memory
    h_A[i] = 1.0f;
  cudaMalloc(&d_A, N*sizeof(float));  // allocate device space for A
  cudaMalloc(&d_sum, sizeof(float));  // allocate device space for sum
  cudaCheckErrors("cudaMalloc failure"); // error checking
  // copy matrix A to device:
  cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //cuda processing sequence step 1 is complete
  atomic_red<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("atomic reduction kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("atomic reduction kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("atomic sum reduction incorrect!\n"); return -1;}
  printf("atomic sum reduction correct!\n");
  const int blocks = 640;
  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //cuda processing sequence step 1 is complete
  reduce_a<<<blocks, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("reduction w/atomic kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("reduction w/atomic kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("reduction w/atomic sum incorrect!\n"); return -1;}
  printf("reduction w/atomic sum correct!\n");
  cudaMemset(d_sum, 0, sizeof(float));
  cudaCheckErrors("cudaMemset failure");
  //cuda processing sequence step 1 is complete
  reduce_ws<<<blocks, BLOCK_SIZE>>>(d_A, d_sum);
  cudaCheckErrors("reduction warp shuffle kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("reduction warp shuffle kernel execution failure or cudaMemcpy H2D failure");
  if (*h_sum != (float)N) {printf("reduction warp shuffle sum incorrect!\n"); return -1;}
  printf("reduction warp shuffle sum correct!\n");
  return 0;
}
  
