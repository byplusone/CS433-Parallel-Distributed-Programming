#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <cstdlib>
using namespace std;

#define N 2048
#define Iteration 100

__global__ void matrixTranspose(int *a, int *b)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y; // row
	int j = blockIdx.x * blockDim.x + threadIdx.x; // col
	int index_in = i*N+j; // (i,j) from matrix A
	int index_out = j*N+i; // transposed index
	b[index_out] = a[index_in];
}

int main(){
	int *a, *b;
	int *d_a, *d_b;
	int size = N*N*sizeof(int);
	clock_t start, end;

	// Alloc space for device copies of a, b
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);

	// Alloc space for host copies of a, b, and setup input values
	a = (int*)malloc(size); 
	b = (int*)malloc(size);
	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++)
			a[i*N+j] = j;

	dim3 grid(64, 64);
	dim3 block(32, 32);

	start = clock();

	// Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	//Launch kernel
	for(int i = 0; i < Iteration; i++)
		matrixTranspose<<<grid, block>>>(d_a, d_b);


	// Copy result back to host
	cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

	end = clock();  

/*	for(int i = 0; i < 10; i++){
		for(int j = 0; j < 10; j++)
			cout<<b[i*N+j]<<" ";
		cout<<endl;
	}*/

	//Cleanup
	free(a); free(b);
	cudaFree(d_a); cudaFree(d_b);

	cout << "Totle Time : " <<(double)(end - start)<< "ms" << endl;

	return 0;
}

