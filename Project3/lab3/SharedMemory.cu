#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <cstdlib>
using namespace std;

#define N 2048
#define Iteration 100

__global__ void matrixTransposeShared(const int *a, int *b)
{
	__shared__ int mat[32][32];
	int bx = blockIdx.x * 32;
	int by = blockIdx.y * 32;
	int i = by + threadIdx.y; int j = bx + threadIdx.x; //input
	int ti = bx + threadIdx.y; int tj = by + threadIdx.x;
	//output
	if(i < N && j < N)
		mat[threadIdx.x][threadIdx.y] = a[i * N + j];
	__syncthreads(); //Wait for all data to be copied
	if(tj < N && ti < N)
		b[ti * N + tj] = mat[threadIdx.y][threadIdx.x];
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
		matrixTransposeShared<<<grid, block>>>(d_a, d_b);


	// Copy result back to host
	cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

	end = clock();  

	//Cleanup
	free(a); free(b);
	cudaFree(d_a); cudaFree(d_b);

	cout << "Totle Time : " <<(double)(end - start)<< "ms" << endl;

	return 0;
}