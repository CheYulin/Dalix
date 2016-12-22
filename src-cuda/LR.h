#include <iostream>
#include <math.h>
#include <sys/timeb.h>
#include "split.h"
#include <string>

using namespace std;

int align_edge_size_compute(int nb_row, int nb_col)
{
	return nb_row * nb_row * nb_col * nb_col - nb_row * nb_row * (1 + nb_col) * nb_col / 2 - nb_col * nb_col * (nb_row + 1) * (nb_row) / 2
		+ (1 + nb_row) * nb_row * (1 + nb_col) * nb_col / 4;
}

int align_edge_iter_compute(int nb_row, int nb_col, int row1, int col1, int row2, int col2)
{
	return row1 * (nb_row - 1) * nb_col * (nb_col - 1) - row1 * (nb_col - 1) * nb_col / 2 * (nb_row - 1) 
		- row1 * (row1 - 1) * nb_col * (nb_col - 1) / 2 + (nb_col - 1) * nb_col * row1 * (row1 - 1) / 4 
		+ (nb_row - 1) * (nb_col - 1) * col1 - (nb_row - 1) * col1 * (col1 - 1) / 2 - row1 * (nb_col - 1) * col1
		+ row1 * col1 * (col1 - 1) / 2 + row2 * (nb_col - col1 - 1) + col2;
}

int lambda_ha_size_compute(int nb_row, int nb_col)
{
	return nb_row * (nb_row - 1) * nb_col / 2;
}

int lambda_ha_iter_compute(int nb_row, int nb_col, int i, int j, int z)
{
	return (i - 1) * i * nb_col / 2 + i * j + z;
}

int lambda_v_size_compute(int nb_row, int nb_col)
{
	return nb_col * (nb_col - 1) * nb_row / 2;
}

int lambda_v_iter_compute(int nb_row, int nb_col, int i, int j, int z)
{
	return (nb_col - 1) * nb_col * i / 2 + (j - 1) * j / 2 + z;
}

__device__ int gpu_align_edge_size_compute(int nb_row, int nb_col)
{
	return nb_row * nb_row * nb_col * nb_col - nb_row * nb_row * (1 + nb_col) * nb_col / 2 - nb_col * nb_col * (nb_row + 1) * (nb_row) / 2
		+ (1 + nb_row) * nb_row * (1 + nb_col) * nb_col / 4;
}

__device__ int gpu_align_edge_iter_compute(int nb_row, int nb_col, int row1, int col1, int row2, int col2)
{
	return row1 * (2 * nb_row - row1 - 1) * nb_col * (nb_col - 1) / 4 
		+ (nb_row - row1 - 1) * col1 * (2 * nb_col - col1 - 1) / 2
		+ row2 * (nb_col - col1 - 1) + col2;
}

__device__ int gpu_lambda_ha_size_compute(int nb_row, int nb_col)
{
	return nb_row * (nb_row - 1) * nb_col / 2;
}

__device__ int gpu_lambda_ha_iter_compute(int nb_row, int nb_col, int i, int j, int z)
{
	return (i - 1) * i * nb_col / 2 + i * j + z;
}

__device__ int gpu_lambda_v_size_compute(int nb_row, int nb_col)
{
	return nb_col * (nb_col - 1) * nb_row / 2;
}

__device__ int gpu_lambda_v_iter_compute(int nb_row, int nb_col, int i, int j, int z)
{
	return (nb_col - 1) * nb_col * i / 2 + (j - 1) * j / 2 + z;
}

__global__ void local_dp2(int nb_row, int nb_col, double * gpu_lambda_a, double * gpu_lambda_h, double * gpu_lambda_v, double * gpu_align_node, int * gpu_dp_node, int * gpu_lambda_for_cik)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	if (bx * 32 + tx < gpu_dp_node[0])
	{
		int row1 = gpu_dp_node[(bx * 32 + tx) * 2 + 1];
		int col1 = gpu_dp_node[(bx * 32 + tx) * 2 + 2];

		double temp3 = 0;
		for (int j = 0; j < row1; j++)
		{
			temp3 += gpu_lambda_h[gpu_lambda_ha_iter_compute(nb_row, nb_col, row1, col1, j)]
				- gpu_lambda_a[gpu_lambda_ha_iter_compute(nb_row, nb_col, row1, col1, j)];
		}
		for (int l = 0; l < col1; l++)
		{
			temp3 += gpu_lambda_v[gpu_lambda_v_iter_compute(nb_row, nb_col, row1, col1, l)];
		}
		int temp4 = gpu_lambda_for_cik[row1 * nb_col + col1 + 1];
		for (int i = gpu_lambda_for_cik[row1 * nb_col + col1]; i < temp4;)
		{
			int r = gpu_lambda_for_cik[i++];
			int s = gpu_lambda_for_cik[i++];
			temp3 -= gpu_lambda_a[gpu_lambda_ha_iter_compute(nb_row, nb_col, r, s, row1)];
		}
		gpu_align_node[row1 * nb_col + col1] += temp3;
	}
}

__global__ void local_dp1(int * gpu_up, int * gpu_lo, int nb_row, int nb_col, double * gpu_align_edge,
	double * gpu_lambda_h, double * gpu_lambda_v, double * gpu_lambda_a, double * gpu_align_node, int * gpu_dp_node, int * gpu_lambda_for_cik)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int row1 = gpu_dp_node[bx * 2 + 1];
	int col1 = gpu_dp_node[bx * 2 + 2];

	__shared__ double dp_mat_test[500];

	dp_mat_test[tx * 2] = 0;
	dp_mat_test[tx * 2 + 1] = 0;

	__syncthreads();

	int align_edge_begin = row1 * (2 * nb_row - row1 - 1) * nb_col * (nb_col - 1) / 4
		+ (nb_row - row1 - 1) * col1 * (2 * nb_col - col1 - 1) / 2;
	
	double replace(0);
	double temp = -10000;
	double current(0);

	for (int i = 0; i < nb_row + nb_col - row1 - col1 - 3; i++)
	{
		if (tx < nb_row - row1 - 1 && tx < i + 1 && i - tx < nb_col - col1 - 1)
		{
			if (row1 + tx + 1 < gpu_up[col1 + 1 + i - tx] && row1 + tx + 1 > gpu_lo[col1 + 1 + i - tx])
			{
				double temp2 = gpu_align_edge[align_edge_begin + tx * (nb_col - col1 - 1) + i - tx];
				temp = replace + temp2 - gpu_lambda_h[gpu_lambda_ha_iter_compute(nb_row, nb_col, row1 + tx + 1, col1 + i - tx + 1, row1)]
					- gpu_lambda_v[gpu_lambda_v_iter_compute(nb_row, nb_col, row1 + tx + 1, col1 + i - tx + 1, col1)];
					//+ (temp2 > 0 ? 0 : gpu_lambda_a[gpu_lambda_ha_iter_compute(nb_row, nb_col, row1 + tx + 1, col1 + i - tx + 1, row1)]);
			}
		}
		__syncthreads();
		if (tx < nb_row - row1 - 1 && tx < i + 1 && i - tx < nb_col - col1 - 1)
		{
			replace = dp_mat_test[i - tx + 1];
			if (dp_mat_test[i - tx] > dp_mat_test[i - tx + 1])
				current = dp_mat_test[i - tx];
			else
				current = dp_mat_test[i - tx + 1];
			if (temp > current)
				current = temp;
		}
		__syncthreads();
		if (tx < nb_row - row1 - 1 && tx < i + 1 && i - tx < nb_col - col1 - 1)
		{
			dp_mat_test[i - tx + 1] = current;
		}
		__syncthreads();
	}
	if (tx == 0)
		gpu_align_node[row1 * nb_col + col1] = dp_mat_test[nb_col - col1 - 1] + 0.2;
}

__global__ void update_g1(int * gpu_up, int * gpu_lo, int nb_row, int nb_col, double * gpu_g_a, double * gpu_g_h, double * gpu_g_v, bool * gpu_align_node_xvalue)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	if (bx * 32 + tx < nb_row * nb_col)
	{
		int row1 = (bx * 32 + tx) / nb_col;
		int col1 = (bx * 32 + tx) % nb_col;

		int temp = gpu_lambda_ha_iter_compute(nb_row, nb_col, row1, col1, 0);
		int temp2 = gpu_lambda_v_iter_compute(nb_row, nb_col, row1, col1, 0);
		for (int j = 0; j < row1; j++)
		{
			if (row1 < gpu_up[col1] && row1 > gpu_lo[col1])
				gpu_g_a[temp] += 1.0;
			if (gpu_align_node_xvalue[row1 * nb_col + col1])
			{
				gpu_g_h[temp] += 1.0;
				gpu_g_a[temp] -= 1.0;
			}
			temp++;
		}
		for (int l = 0; l < col1; l++)
		{
			if (gpu_align_node_xvalue[row1 * nb_col + col1])
				gpu_g_v[temp2] += 1.0;
			temp2++;
		}
	}
}

__global__ void update_g2(int nb_row, int nb_col, double * gpu_g_a, int * gpu_lambda_for_cik, int * gpu_align_node_xvalue_short)
{
	int row1 = gpu_align_node_xvalue_short[blockIdx.x * 2 + 1];
	int col1 = gpu_align_node_xvalue_short[blockIdx.x * 2 + 2];
	int tx = threadIdx.x;
	for (int i = gpu_lambda_for_cik[row1 * nb_col + col1] + tx * 2; i < gpu_lambda_for_cik[row1 * nb_col + col1 + 1]; i += 512)
	{
		int r = gpu_lambda_for_cik[i];
		int s = gpu_lambda_for_cik[i + 1];
		gpu_g_a[gpu_lambda_ha_iter_compute(nb_row, nb_col, r, s, row1)] -= 1.0;
	}
}

__global__ void update_g3(int * gpu_up, int * gpu_lo, int nb_row, int nb_col, int * gpu_align_node_xvalue_short,
	double * gpu_align_edge, double * gpu_lambda_h, double * gpu_lambda_v, double * gpu_lambda_a, double * gpu_g_h, double * gpu_g_v, double * gpu_g_a, int * gpu_path_mat)
{
	int bx = blockIdx.x;
	int row1 = gpu_align_node_xvalue_short[bx * 2 + 1];
	int col1 = gpu_align_node_xvalue_short[bx * 2 + 2];
	int tx = threadIdx.x;

	__shared__ double dp_mat_test[500];

	dp_mat_test[tx * 2] = 0;
	dp_mat_test[tx * 2 + 1] = 0;

	__syncthreads();

	int align_edge_begin = row1 * (2 * nb_row - row1 - 1) * nb_col * (nb_col - 1) / 4
		+ (nb_row - row1 - 1) * col1 * (2 * nb_col - col1 - 1) / 2 + tx * (nb_col - col1 - 1) - tx;

	double replace(0);
	double temp = -10000;
	double current(0);

	for (int i = 0; i < nb_row + nb_col - row1 - col1 - 3; i++)
	{
		if (tx < nb_row - row1 - 1 && tx < i + 1 && i - tx < nb_col - col1 - 1)
		{
			if (row1 + tx + 1 < gpu_up[col1 + 1 + i - tx] && row1 + tx + 1 > gpu_lo[col1 + 1 + i - tx])
			{
				double temp2 = gpu_align_edge[align_edge_begin + i];
				temp = replace + temp2 - gpu_lambda_h[gpu_lambda_ha_iter_compute(nb_row, nb_col, row1 + tx + 1, col1 + i - tx + 1, row1)]
					- gpu_lambda_v[gpu_lambda_v_iter_compute(nb_row, nb_col, row1 + tx + 1, col1 + i - tx + 1, col1)];
			}
		}
		__syncthreads();
		if (tx < nb_row - row1 - 1 && tx < i + 1 && i - tx < nb_col - col1 - 1)
		{
			replace = dp_mat_test[i - tx + 1];
			if (dp_mat_test[i - tx] > dp_mat_test[i - tx + 1])
			{
				current = dp_mat_test[i - tx];
				gpu_path_mat[bx * 200 * 200 + (tx + 1) * 200 + i - tx + 1] = -1;
			}
			else
			{
				current = dp_mat_test[i - tx + 1];
				gpu_path_mat[bx * 200 * 200 + (tx + 1) * 200 + i - tx + 1] = -2;
			}
			if (temp > current)
			{
				current = temp;
				gpu_path_mat[bx * 200 * 200 + (tx + 1) * 200 + i - tx + 1] = 1;
			}
		}
		__syncthreads();
		if (tx < nb_row - row1 - 1 && tx < i + 1 && i - tx < nb_col - col1 - 1)
		{
			dp_mat_test[i - tx + 1] = current;
		}
		__syncthreads();
	}
}

__global__ void update_g4(int * gpu_up, int * gpu_lo, int nb_row, int nb_col, int * gpu_align_node_xvalue_short,
	double * gpu_align_edge, double * gpu_lambda_h, double * gpu_lambda_v, double * gpu_lambda_a, double * gpu_g_h, double * gpu_g_v, double * gpu_g_a, int * gpu_path_mat)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	if (bx * 32 + tx < gpu_align_node_xvalue_short[0])
	{
		int row1 = gpu_align_node_xvalue_short[(bx * 32 + tx) * 2 + 1];
		int col1 = gpu_align_node_xvalue_short[(bx * 32 + tx) * 2 + 2];

		int x = nb_row - row1;
		int y = nb_col - col1;
		while (x != 1 || y != 1)
		{
			if (gpu_path_mat[(bx * 32 + tx) * 200 * 200 + (x - 1) * 200 + y - 1] == -1)
				y--;
			else if (gpu_path_mat[(bx * 32 + tx) * 200 * 200 + (x - 1) * 200 + y - 1] == -2)
				x--;
			else if (gpu_path_mat[(bx * 32 + tx) * 200 * 200 + (x - 1) * 200 + y - 1] == 1)
			{
				gpu_g_h[gpu_lambda_ha_iter_compute(nb_row, nb_col, row1 + x - 1, col1 + y - 1, row1)] -= 1.0;
				gpu_g_v[gpu_lambda_v_iter_compute(nb_row, nb_col, row1 + x - 1, col1 + y - 1, col1)] -= 1.0;
				if (gpu_align_edge[gpu_align_edge_iter_compute(nb_row, nb_col, row1, col1, x - 2, y - 2)] <= 0)
					gpu_g_a[gpu_lambda_ha_iter_compute(nb_row, nb_col, row1 + x - 1, col1 + y - 1, row1)] += 1.0;
				x--; y--;
			}
		}
	}
}

__global__ void compute_thelta(int * gpu_up, int * gpu_lo, int nb_row, int nb_col, double * gpu_g_h, double * gpu_g_v, double * gpu_g_a, int * temp, double * gpu_lambda_a)
{
	int id = blockIdx.x * 128 + threadIdx.x;
	int row1 = id / nb_col;
	int col1 = id % nb_col;

	if (row1 < nb_row && row1 < gpu_up[col1] && row1 > gpu_lo[col1])
	{
		for (int j = gpu_lambda_ha_iter_compute(nb_row, nb_col, row1, col1, 0); j < gpu_lambda_ha_iter_compute(nb_row, nb_col, row1, col1, 0) + row1; j++)
		{
			/*if (gpu_g_h[j] != 0)
				temp[id]++;*/
			if (gpu_g_h[j] == -1)
				temp[id]++;
			if (gpu_g_h[j] == 1 && gpu_lambda_a[j] > 0)
				temp[id]++;
			if (gpu_g_a[j] != 0)
				temp[id]++;
		}
		for (int l = gpu_lambda_v_iter_compute(nb_row, nb_col, row1, col1, 0); l < gpu_lambda_v_iter_compute(nb_row, nb_col, row1, col1, 0) + col1; l++)
		{
			if (gpu_g_v[l] != 0)
				temp[id]++;
		}
	}
}

__global__ void update_lambda(int * gpu_up, int * gpu_lo, int nb_row, int nb_col, double * gpu_g_h, double * gpu_g_v, double * gpu_g_a, double * gpu_lambda_h, double * gpu_lambda_v, double * gpu_lambda_a, double thelta)
{
	int id = blockIdx.x;
	int tx = threadIdx.x;
	int row1 = id / nb_col;
	int col1 = id % nb_col;

	if (row1 < nb_row && row1 < gpu_up[col1] && row1 > gpu_lo[col1])
	{
		for (int j = gpu_lambda_ha_iter_compute(nb_row, nb_col, row1, col1, 0) + (row1 / 64 + 1) * tx; j < gpu_lambda_ha_iter_compute(nb_row, nb_col, row1, col1, 0) + (row1 / 64 + 1) * (tx + 1); j++)
		{
			if (j < gpu_lambda_ha_iter_compute(nb_row, nb_col, row1, col1, 0) + row1)
			{
				gpu_lambda_h[j] -= thelta * gpu_g_h[j];
				gpu_g_h[j] = 0;
				if (gpu_lambda_h[j] < 0)
					gpu_lambda_h[j] = 0;

				gpu_lambda_a[j] -= thelta * gpu_g_a[j];
				gpu_g_a[j] = 0;
				if (gpu_lambda_a[j] < 0)
					gpu_lambda_a[j] = 0;
			}
		}
		for (int l = gpu_lambda_v_iter_compute(nb_row, nb_col, row1, col1, 0) + (col1 / 64 + 1) * tx; l < gpu_lambda_v_iter_compute(nb_row, nb_col, row1, col1, 0) + (col1 / 64 + 1) * (tx + 1); l++)
		{
			if (l < gpu_lambda_v_iter_compute(nb_row, nb_col, row1, col1, 0) + col1)
			{
				gpu_lambda_v[l] -= thelta * gpu_g_v[l];
				gpu_g_v[l] = 0;
				if (gpu_lambda_v[l] < 0)
					gpu_lambda_v[l] = 0;
			}
		}
	}
}

int solve_lr(int nb_row, int nb_col, double * gpu_align_edge, int * cpu_lambda_for_cik, problem * current_problem, int cpu_lambda_for_cik_size, int nb_iteration, int * alignment, int precise, 
	double global_lb1)
{
	double global_ub(INFINITY);
	double global_lb(-INFINITY);

	double alpha(1);
	int nb_improve(0);
	int nb_noimprove(0);
	double last_ub(0);
	double last_lb(0);
	int alpha_change(0);
	bool need_to_change[200][200] = { 0 };
	memset(need_to_change, 1, 200 * 200 * sizeof(bool));

	//struct timeb startTime, endTime;
	//ftime(&startTime);

	int * gpu_up;
	int * gpu_lo;

	cudaMalloc((void**)&gpu_up, sizeof(int) * nb_col);
	cudaMalloc((void**)&gpu_lo, sizeof(int) * nb_col);

	cudaMemcpy(gpu_up, current_problem->up, sizeof(int) * nb_col, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_lo, current_problem->lo, sizeof(int) * nb_col, cudaMemcpyHostToDevice);

	//ftime(&startTime);

	int align_edge_size = align_edge_size_compute(nb_row, nb_col);
	double * cpu_align_edge = new double[align_edge_size];

	cudaMemcpy(cpu_align_edge, gpu_align_edge, sizeof(double) * align_edge_size, cudaMemcpyDeviceToHost);
    
    double * cpu_lambda_h = new double[lambda_ha_size_compute(nb_row, nb_col)];
    double * cpu_lambda_a = new double[lambda_ha_size_compute(nb_row, nb_col)];
    double * cpu_lambda_v = new double[lambda_v_size_compute(nb_row, nb_col)];
	double * cpu_g_h = new double[lambda_ha_size_compute(nb_row, nb_col)];
	double * cpu_g_a = new double[lambda_ha_size_compute(nb_row, nb_col)];
	double * cpu_g_v = new double[lambda_v_size_compute(nb_row, nb_col)];

	for (int i = 0; i < lambda_ha_size_compute(nb_row, nb_col); i++)
    {
         cpu_lambda_h[i] = 0;
         cpu_lambda_a[i] = 0;
    }

	for (int i = 0; i < lambda_v_size_compute(nb_row, nb_col); i++)
	{
		 cpu_lambda_v[i] = 0;
	}

	for (int i = 0; i < lambda_ha_size_compute(nb_row, nb_col); i++)
	{
		cpu_g_h[i] = 0;
		cpu_g_a[i] = 0;
	}

	for (int i = 0; i < lambda_v_size_compute(nb_row, nb_col); i++)
	{
		cpu_g_v[i] = 0;
	}

	double * gpu_lambda_h;
	double * gpu_lambda_v;
	double * gpu_lambda_a;

	cudaMalloc((void**)&gpu_lambda_h, sizeof(double) * lambda_ha_size_compute(nb_row, nb_col));
	cudaMalloc((void**)&gpu_lambda_v, sizeof(double) * lambda_v_size_compute(nb_row, nb_col));
	cudaMalloc((void**)&gpu_lambda_a, sizeof(double) * lambda_ha_size_compute(nb_row, nb_col));

	cudaMemcpy(gpu_lambda_h, cpu_lambda_h, sizeof(double) * lambda_ha_size_compute(nb_row, nb_col), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_lambda_v, cpu_lambda_v, sizeof(double) * lambda_v_size_compute(nb_row, nb_col), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_lambda_a, cpu_lambda_a, sizeof(double) * lambda_ha_size_compute(nb_row, nb_col), cudaMemcpyHostToDevice);

	double * gpu_g_h;
	double * gpu_g_v;
	double * gpu_g_a;

	cudaMalloc((void**)&gpu_g_h, sizeof(double) * lambda_ha_size_compute(nb_row, nb_col));
	cudaMalloc((void**)&gpu_g_v, sizeof(double) * lambda_v_size_compute(nb_row, nb_col));
	cudaMalloc((void**)&gpu_g_a, sizeof(double) * lambda_ha_size_compute(nb_row, nb_col));

	cudaMemcpy(gpu_g_h, cpu_g_h, sizeof(double) * lambda_ha_size_compute(nb_row, nb_col), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_g_v, cpu_g_v, sizeof(double) * lambda_v_size_compute(nb_row, nb_col), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_g_a, cpu_g_a, sizeof(double) * lambda_ha_size_compute(nb_row, nb_col), cudaMemcpyHostToDevice);

	int * gpu_lambda_for_cik;
	cudaMalloc((void**)&gpu_lambda_for_cik, sizeof(int) * (cpu_lambda_for_cik_size + nb_row * nb_col));
	cudaMemcpy(gpu_lambda_for_cik, cpu_lambda_for_cik, sizeof(int) * (cpu_lambda_for_cik_size + nb_row * nb_col), cudaMemcpyHostToDevice);

	double * cpu_align_node = new double [nb_row * nb_col];
	for (int i = 0; i < nb_row * nb_col; i++)
		cpu_align_node[i] = -INFINITY;

	double * gpu_align_node;
	cudaMalloc((void**)&gpu_align_node, sizeof(double) * nb_row * nb_col);
	cudaMemcpy(gpu_align_node, cpu_align_node, sizeof(double) * nb_row * nb_col, cudaMemcpyHostToDevice);

	int * gpu_align_node_xvalue_short;
	cudaMalloc((void**)&gpu_align_node_xvalue_short, sizeof(int)* 400);

	int * gpu_dp_node;
	cudaMalloc((void**)&gpu_dp_node, sizeof(int) * 80000);

	int * path_mat = new int[200 * 200 * 200];
	for (int i = 0; i < 200; i++)
	{
		for (int j = 0; j < 200; j++)
		{
			for (int k = 0; k < 200; k++)
			{
				if (j == 0)
					path_mat[i * 200 * 200 + j * 200 + k] = -1;
				else if (k == 0)
					path_mat[i * 200 * 200 + j * 200 + k] = -2;
				else
					path_mat[i * 200 * 200 + j * 200 + k] = 0;
			}
		}
	}

	int * gpu_path_mat;
	cudaMalloc((void**)&gpu_path_mat, sizeof(int) * 200 * 200 * 200);

	int * gpu_temp;
	cudaMalloc((void**)&gpu_temp, sizeof(int) * 40000);

	//ftime(&endTime);
	//cout << "gpu_initialization: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << endl;

	for (int iteration = 0; iteration < nb_iteration; iteration++)
	{
		cudaMemcpy(gpu_path_mat, path_mat, sizeof(int) * 200 * 200 * 200, cudaMemcpyHostToDevice);
		int cpu_temp[40000] = { 0 };
		cudaMemcpy(gpu_temp, cpu_temp, sizeof(int) * 40000, cudaMemcpyHostToDevice);

		double ub(0);
		double lb(0);

		double * gpu_ub;
		cudaMalloc(&gpu_ub, sizeof(double));

		double * gpu_lb;
		cudaMalloc(&gpu_lb, sizeof(double));
		bool * cpu_align_node_xvalue = new bool[nb_row * nb_col];
		for (int i = 0; i < nb_row * nb_col; i++)
			cpu_align_node_xvalue[i] = 0;

		bool * gpu_align_node_xvalue;
		cudaMalloc((void**)&gpu_align_node_xvalue, sizeof(bool)* nb_row * nb_col);
		cudaMemcpy(gpu_align_node_xvalue, cpu_align_node_xvalue, sizeof(bool)* nb_row * nb_col, cudaMemcpyHostToDevice);

		int * dp_node = new int[80000];
		int dp_node_iter = 1;
		int dp_node_num = 0;
		for (int row1 = 0; row1 < nb_row; row1++)
		{
			for (int col1 = 0; col1 < nb_col; col1++)
			{
				if (row1 < current_problem->up[col1] && row1 > current_problem->lo[col1])
				{
					if (need_to_change[row1][col1])
					{
						dp_node[dp_node_iter++] = row1;
						dp_node[dp_node_iter++] = col1;
						dp_node_num++;
					}
				}
			}
		}
		dp_node[0] = dp_node_num;

		cudaMemcpy(gpu_dp_node, dp_node, sizeof(int) * 80000, cudaMemcpyHostToDevice);

		cudaFuncSetCacheConfig(local_dp1, cudaFuncCachePreferL1);

		//ftime(&startTime);
		local_dp1 << <dp_node_num, nb_row >> >(gpu_up, gpu_lo, nb_row, nb_col, gpu_align_edge, gpu_lambda_h, gpu_lambda_v, gpu_lambda_a, gpu_align_node, gpu_dp_node, gpu_lambda_for_cik);
		local_dp2 << <dp_node_num / 32 + 1, 32 >> >(nb_row, nb_col, gpu_lambda_a, gpu_lambda_h, gpu_lambda_v, gpu_align_node, gpu_dp_node, gpu_lambda_for_cik);

		//cudaThreadSynchronize();
		//ftime(&endTime);
		//cout << "local_dp: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << endl;
		//ftime(&startTime);

		cudaMemcpy(cpu_align_node, gpu_align_node, sizeof(double)* nb_row * nb_col, cudaMemcpyDeviceToHost);

		double dp_mat[200][200] = { 0 };
		int path_mat_global[200][200] = { 0 };
		for (int i = 0; i < nb_row; i++)
		{
			if (i < current_problem->up[0] && i > current_problem->lo[0])
			{
				dp_mat[i][0] = cpu_align_node[i * nb_col]; path_mat_global[i][0] = -2;
			}
			else
			{
				dp_mat[i][0] = -INFINITY; path_mat_global[i][0] = -2;
			}
		}
		for (int i = 0; i < nb_col; i++)
		{
			if (0 < current_problem->up[i] && 0 > current_problem->lo[i])
			{
				dp_mat[0][i] = cpu_align_node[i]; path_mat_global[0][i] = -1;
			}
			else
			{
				dp_mat[0][i] = -INFINITY; path_mat_global[0][i] = -1;
			}
		}

		for (int row = 1; row < nb_row; row++)
		{
			for (int col = 1; col < nb_col; col++)
			{
				dp_mat[row][col] = dp_mat[row][col - 1];
				path_mat_global[row][col] = -1;

				double temp = dp_mat[row - 1][col];
				if (temp > dp_mat[row][col])
				{
					dp_mat[row][col] = temp;
					path_mat_global[row][col] = -2;
				}

				if (need_to_change[row][col])
				{
					temp = dp_mat[row - 1][col - 1] + cpu_align_node[row * nb_col + col];
					if (temp > dp_mat[row][col])
					{
						dp_mat[row][col] = temp;
						path_mat_global[row][col] = 1;
					}
				}
			}
		}

		ub = dp_mat[nb_row - 1][nb_col - 1]; 

		int x_1 = nb_row;
		int y_1 = nb_col;
		while (x_1 != 1 || y_1 != 1)
		{
			if (path_mat_global[x_1 - 1][y_1 - 1] == -1)
				y_1--;
			else if (path_mat_global[x_1 - 1][y_1 - 1] == -2)
				x_1--;
			else if (path_mat_global[x_1 - 1][y_1 - 1] == 1)
			{
				cpu_align_node_xvalue[(x_1 - 1) * nb_col + y_1 - 1] = true; x_1--; y_1--;
				if (x_1 - 1 == 0 || y_1 - 1 == 0)
					cpu_align_node_xvalue[(x_1 - 1) * nb_col + y_1 - 1] = 1;
			}
		}
		cudaMemcpy(gpu_align_node_xvalue, cpu_align_node_xvalue, sizeof(bool) * nb_row * nb_col, cudaMemcpyHostToDevice);

		int * cpu_align_node_xvalue_short = new int[400];
		int xvalue_iter = 1;
		for (int row1 = 0; row1 < nb_row; row1++)
		{
			for (int col1 = 0; col1 < nb_col; col1++)
			{
				if (cpu_align_node_xvalue[row1 * nb_col + col1])
				{
					cpu_align_node_xvalue_short[xvalue_iter++] = row1;
					cpu_align_node_xvalue_short[xvalue_iter++] = col1;
				}
			}
		}
		cpu_align_node_xvalue_short[0] = (xvalue_iter - 1) / 2;
		cudaMemcpy(gpu_align_node_xvalue_short, cpu_align_node_xvalue_short, sizeof(int) * 400, cudaMemcpyHostToDevice);

		//--------------------------------------------compute lower bound
		for (int row1 = 0; row1 < nb_row; row1++)
		{
			for (int col1 = 0; col1 < nb_col; col1++)
			{
				if (cpu_align_node_xvalue[row1 * nb_col + col1])
				{
					lb += 0.2;
					for (int row2 = 0; row2 < nb_row - row1 - 1; row2++)
					{
						for (int col2 = 0; col2 < nb_col - col1 - 1; col2++)
						{
							if (cpu_align_node_xvalue[(row1 + row2 + 1) * nb_col + col1 + col2 + 1])
								lb += cpu_align_edge[align_edge_iter_compute(nb_row, nb_col, row1, col1, row2, col2)];
						}
					}
				}
			}
		}
		//--------------------------------------------compute lower bound
		if (ub < global_lb1 && global_lb < global_lb1)
			return 2;

		if (ub < global_ub)
			global_ub = ub;
		if (lb > global_lb)
		{
			global_lb = lb;
			for (int i = 0; i < nb_row; i++)
				alignment[i] = -1;
			for (int i = 0; i < cpu_align_node_xvalue_short[0]; i++)
			{
				alignment[cpu_align_node_xvalue_short[i * 2 + 1]] = cpu_align_node_xvalue_short[i * 2 + 2];
			}
		}

		//cout << "ub: " << ub << endl;
		//cout << "lb: " << lb << endl;

		if (iteration > 0)
		{
			if (ub < last_ub && lb > last_lb)
			{
				nb_improve++;
				nb_noimprove = 0;
				if (nb_improve >= 5)
				{
					alpha /= 0.9;
					nb_improve = 0;
				}
			}
			else //if (ub >= last_ub && lb <= last_lb)
			{
				nb_improve = 0;
				nb_noimprove++;
				if (lb <= last_lb)
				{
					alpha_change++;
					if (alpha_change == 1)
						alpha = 1;
				}
				if (nb_noimprove >= 5)
					alpha *= 0.9;
			}
		}

		last_ub = ub;
		last_lb = lb;

		if (global_ub <= global_lb)
		{
			cout << "iteration: " << iteration << endl;
			cout << "final_solution: " << global_lb << endl;
			return 1;
		}
		//---------------------------------------------------------------------lambda update
		double thelta(0);
		memset(need_to_change, 0, 200 * 200 * sizeof(bool));

		for (int row1 = 0; row1 < nb_row; row1++)
		{
			for (int col1 = 0; col1 < nb_col; col1++)
			{
				if (cpu_align_node_xvalue[row1 * nb_col + col1])
				{
					for (int x = - precise / 2; x < 1 + precise / 2; x++)
					{
						if (row1 + x >= 0 && row1 + x < nb_row)
							need_to_change[row1 + x][col1] = true;
						if (col1 + x >= 0 && col1 + x < nb_col)
							need_to_change[row1][col1 + x] = true;
					}
				}
			}
		}

		update_g1 << <nb_row * nb_col / 32 + 1, 32 >> >(gpu_up, gpu_lo, nb_row, nb_col, gpu_g_a, gpu_g_h, gpu_g_v, gpu_align_node_xvalue);

		update_g2 << <cpu_align_node_xvalue_short[0], 256 >> >(nb_row, nb_col, gpu_g_a, gpu_lambda_for_cik, gpu_align_node_xvalue_short);

		update_g3 << <cpu_align_node_xvalue_short[0], nb_row >> >(gpu_up, gpu_lo, nb_row, nb_col, gpu_align_node_xvalue_short,
			gpu_align_edge, gpu_lambda_h, gpu_lambda_v, gpu_lambda_a, gpu_g_h, gpu_g_v, gpu_g_a, gpu_path_mat);

		update_g4 << <cpu_align_node_xvalue_short[0] / 32 + 1, 32 >> >(gpu_up, gpu_lo, nb_row, nb_col, gpu_align_node_xvalue_short,
			gpu_align_edge, gpu_lambda_h, gpu_lambda_v, gpu_lambda_a, gpu_g_h, gpu_g_v, gpu_g_a, gpu_path_mat);

		compute_thelta << <nb_row * nb_col / 128 + 1, 128 >> >(gpu_up, gpu_lo, nb_row, nb_col, gpu_g_h, gpu_g_v, gpu_g_a, gpu_temp, gpu_lambda_a);
		cudaMemcpy(cpu_temp, gpu_temp, sizeof(int) * 40000, cudaMemcpyDeviceToHost);
		for (int i = 1; i < nb_row * nb_col; i++)
			cpu_temp[0] += cpu_temp[i];

		thelta = alpha * (ub - global_lb) / cpu_temp[0];
		update_lambda << <nb_row * nb_col, 64 >> >(gpu_up, gpu_lo, nb_row, nb_col, gpu_g_h, gpu_g_v, gpu_g_a, gpu_lambda_h, gpu_lambda_v, gpu_lambda_a, thelta);

		//cudaThreadSynchronize();
		//ftime(&endTime);
		//cout << "update lambda: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << endl << endl;
		//ftime(&startTime);

		delete[] cpu_align_node_xvalue;
		delete[] cpu_align_node_xvalue_short;
		delete[] dp_node;

		cudaFree(gpu_align_node_xvalue);
		cudaFree(gpu_ub);
		cudaFree(gpu_lb);
	}
	//---------------------------------------------------------------------LR
	delete[] cpu_lambda_h;
	delete[] cpu_lambda_a;
	delete[] cpu_lambda_v;
	delete[] cpu_g_h;
	delete[] cpu_g_a;
	delete[] cpu_g_v;
	delete[] path_mat;
	delete[] cpu_align_edge;
	delete[] cpu_align_node;

	current_problem->ub = global_ub;
	current_problem->lb = global_lb;

	cudaFree(gpu_up);
	cudaFree(gpu_lo);
	//cudaFree(gpu_align_edge);
	cudaFree(gpu_lambda_h);
	cudaFree(gpu_lambda_v);
	cudaFree(gpu_lambda_a);
	cudaFree(gpu_g_h);
	cudaFree(gpu_g_v);
	cudaFree(gpu_g_a);
	cudaFree(gpu_lambda_for_cik);
	cudaFree(gpu_align_node);
	cudaFree(gpu_align_node_xvalue_short);
	cudaFree(gpu_dp_node);
	cudaFree(gpu_path_mat);
	cudaFree(gpu_temp);

	return 0;
}