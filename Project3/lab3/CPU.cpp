#include <time.h>
#include <iostream>
#include <cstdlib>
using namespace std;
#define N 2048
#define Iteration 100

int main(){
	int **transpose, **matrix;

	clock_t start, end;

	transpose = (int**)malloc(sizeof(int*)*N);
	for(int i = 0; i < N; i++)
		transpose[i] = (int*)malloc(sizeof(int)*N);
	matrix = (int**)malloc(sizeof(int*)*N);
	for(int i = 0; i < N; i++)
		matrix[i] = (int*)malloc(sizeof(int)*N);

	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++)
			matrix[i][j] = j;

	start = clock();
	for(int k = 0; k < Iteration; k++)
		for(int i = 0; i < N; i++)
			for(int j = 0; j < N; j++)
				transpose[i][j] = matrix[j][i];
	end = clock();

	cout << "Totle Time : " <<(double)(end - start)<< "ms" << endl;
	system("pause");
	return 0;
}