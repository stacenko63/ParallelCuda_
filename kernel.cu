#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;
#define BLOCK_SIZE 16
__global__ void multiplicate_square_matrix(int* d_a, int* d_b, int* d_result, int n)
{
	__shared__ int tile_a[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int tile_b[BLOCK_SIZE][BLOCK_SIZE];
	int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	int tmp = 0;
	int idx;
	for (int sub = 0; sub < gridDim.x; ++sub) {
		idx = row * n + sub * BLOCK_SIZE + threadIdx.x;
		if (idx >= n * n) {
			tile_a[threadIdx.y][threadIdx.x] = 0;
		}
		else {
			tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
		}
		idx = (sub * BLOCK_SIZE + threadIdx.y) * n + col;
		if (idx >= n * n) {
			tile_b[threadIdx.y][threadIdx.x] = 0;
		}
		else {
			tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
		}
		__syncthreads();
		for (int k = 0; k < BLOCK_SIZE; ++k) {
			tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
		}
		__syncthreads();
	}
	if (row < n && col < n) {
		d_result[row * n + col] = tmp;
	}
}
__global__ void multiplicate_matrix(int* a, int* b, int* c, int m, int n, int k) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int sum = 0;
	if (col < k && row < m) {
		for (int i = 0; i < n; i++) {
			sum += a[row * n + i] * b[i * k + col];
		}
		c[row * k + col] = sum;
	}
}
__global__ void transpose_matrix(int* mat_in, int* mat_out, unsigned int rows, unsigned int cols) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < cols && idy < rows) {
		unsigned int pos = idy * cols + idx;
		unsigned int trans_pos = idx * rows + idy;
		mat_out[trans_pos] = mat_in[pos];
	}
}
void print_matrix(int* matrix, int N, const string& filename) {
	ofstream fout;
	fout.open("matrix/" + to_string(N) + "/" + filename);
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++) 
			fout << matrix[i * N + j] << ' ';
		fout << '\n';
	}
	fout.close();
}
void generate_matrix(int* matrix, int m, int n) {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			matrix[i * n + j] = rand() % 100;
		}
	}
}
int main(int argc, char const* argv[]) {
	int start_size = 500, end_size = 900;
	for (int i = start_size; i <= end_size; i += 100) {
		int m = i, n = i, k = i;
		srand(time(NULL));
		int* first_matrix, *second_matrix, *result_matrix;
		cudaMallocHost((void**)&first_matrix, sizeof(int) * m * n);
		cudaMallocHost((void**)&second_matrix, sizeof(int) * n * k);
		cudaMallocHost((void**)&result_matrix, sizeof(int) * m * k);
		generate_matrix(first_matrix, m, n);
		generate_matrix(second_matrix, m, n);
		print_matrix(first_matrix, n, "A.txt");
		print_matrix(second_matrix, n, "B.txt");
		float gpu_elapsed_time_ms;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
		int* d_a, * d_b, * d_c;
		cudaMalloc((void**)&d_a, sizeof(int) * m * n);
		cudaMalloc((void**)&d_b, sizeof(int) * n * k);
		cudaMalloc((void**)&d_c, sizeof(int) * m * k);
		cudaMemcpy(d_a, first_matrix, sizeof(int) * m * n, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, second_matrix, sizeof(int) * n * k, cudaMemcpyHostToDevice);
		unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
		unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
		dim3 dimGrid(grid_cols, grid_rows);
		dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
		multiplicate_square_matrix << <dimGrid, dimBlock >> > (d_a, d_b, d_c, n);
		cudaMemcpy(result_matrix, d_c, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();
		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
		cout << m << " * " << n << "on GPU : "<< gpu_elapsed_time_ms / 1000 << "\n\n";
		print_matrix(result_matrix, n, "C.txt");
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		cudaFreeHost(first_matrix);
		cudaFreeHost(second_matrix);
		cudaFreeHost(result_matrix);
	}
	return 0;
}
