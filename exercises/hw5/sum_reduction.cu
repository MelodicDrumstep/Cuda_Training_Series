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