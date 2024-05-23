/*

This version is a portable vector addition code 
that can be run both on cpu and gpu using hemi library

*/


#include <hemi/hemi.h>
#include <hemi/parallel_for.h>
#include <stdio.h>
#include <stdlib.h>

const int DSIZE = 32*1048424;

// vector add function: C = A + B
void vadd(const float *A, const float *B, float *C, int ds)
{
  hemi::parallel_for(0, ds, [=] HEMI_LAMBDA (int idx) {
    C[idx] = A[idx] + B[idx];
  });
}

int main(){

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
  h_A = new float[DSIZE];  // allocate space for vectors in host memory
  h_B = new float[DSIZE];
  h_C = new float[DSIZE];
  for (int i = 0; i < DSIZE; i++){  // initialize vectors in host memory
    h_A[i] = rand()/(float)RAND_MAX;
    h_B[i] = rand()/(float)RAND_MAX;
    h_C[i] = 0;
  }

  d_A = hemi::device_malloc<float>(DSIZE);  // allocate device space for vector A
  d_B = hemi::device_malloc<float>(DSIZE);  // allocate device space for vector B
  d_C = hemi::device_malloc<float>(DSIZE);  // allocate device space for vector C

  hemi::copy(d_A, h_A, DSIZE * sizeof(float));  // copy vector A to device
  hemi::copy(d_B, h_B, DSIZE * sizeof(float));  // copy vector B to device

  vadd(d_A, d_B, d_C, DSIZE);  // perform vector addition on device

  hemi::copy(h_C, d_C, DSIZE * sizeof(float));  // copy vector C from device to host

  printf("A[0] = %f\n", h_A[0]);
  printf("B[0] = %f\n", h_B[0]);
  printf("C[0] = %f\n", h_C[0]);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  hemi::device_free(d_A);
  hemi::device_free(d_B);
  hemi::device_free(d_C);

  return 0;
}
