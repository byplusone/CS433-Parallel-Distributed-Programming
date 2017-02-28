#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <cstdlib>
using namespace std;

#define N 2048
#define Iteration 100

const int TILE = 32; const int SIDE = 8;
__global__ void matrixTransposeUnrolled(const int *a, int *b)
{
	__shared__ int mat[TILE][TILE + 1];
	int x = blockIdx.x * TILE + threadIdx.x;
	int y = blockIdx.y * TILE + threadIdx.y;

	#pragma unroll TILE/SIDE
	for(int k = 0; k < TILE ; k += SIDE) {
		if(x < N && y + k < N)
			mat[threadIdx.y + k][threadIdx.x] = a[((y + k) * N) + x];
	}

	__syncthreads();

	x = blockIdx.y * TILE + threadIdx.x;
	y = blockIdx.x * TILE + threadIdx.y;

	#pragma unroll TILE/SIDE
	for(int k = 0; k < TILE; k += SIDE) {
		if(x < N && y + k < N)
			b[(y + k) * N + x] = mat[threadIdx.x][threadIdx.y + k];
	}
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
	dim3 block(32, 8);

	start = clock();

	// Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	//Launch kernel
	for(int i = 0; i < Iteration; i++)
		matrixTransposeUnrolled<<<grid, block>>>(d_a, d_b);


	// Copy result back to host
	cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

	end = clock();  

	for(int i = 0; i < 10; i++){
		for(int j = 0; j < 10; j++)
			cout<<b[i*N+j]<<" ";
		cout<<endl;
	}

	//Cleanup
	free(a); free(b);
	cudaFree(d_a); cudaFree(d_b);

	cout << "Totle Time : " <<(double)(end - start)<< "ms" << endl;

	return 0;
}