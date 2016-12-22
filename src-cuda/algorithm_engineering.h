#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <vector>
#include <math.h>
#include <string>
#include <sys/timeb.h>
#include <ctime>

using namespace std;

__global__ void local_dp_1(int nb_row, int nb_col, double * gpu_align_edge, double * dp_node_value)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	if (bx * 64 + tx < nb_row * nb_col)
	{
		int row1 = (bx * 64 + tx) / nb_col;
		int col1 = (bx * 64 + tx) % nb_col;

		int align_edge_begin = row1 * (2 * nb_row - row1 - 1) * nb_col * (nb_col - 1) / 4
			+ (nb_row - row1 - 1) * col1 * (2 * nb_col - col1 - 1) / 2 - nb_col + col1;

		double dp_mat_test[200] = { 0 };
		double replace(0);

		dp_node_value[row1 * nb_col + col1] = 0;
		for (int row = row1 + 1; row < nb_row; row++)
		{
			replace = 0;
			for (int col = col1 + 1; col < nb_col; col++)
			{
				double temp = replace + gpu_align_edge[align_edge_begin + (row - row1) * (nb_col - col1 - 1) + col - col1];
				replace = dp_mat_test[col];

				if (dp_mat_test[col] < dp_mat_test[col - 1])
					dp_mat_test[col] = dp_mat_test[col - 1];
				if (temp > dp_mat_test[col])
					dp_mat_test[col] = temp;
			}
		}
		dp_node_value[row1 * nb_col + col1] = dp_mat_test[nb_col - 1] + 0.2;
	}
}

__global__ void local_dp_2(int nb_row, int nb_col, double * dp_node_value, int * gpu_domain, double dali_score)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	if (bx * 64 + tx < nb_row * nb_col)
	{
		int row1 = (bx * 64 + tx) / nb_col;
		int col1 = (bx * 64 + tx) % nb_col;

		gpu_domain[row1 * nb_col + col1] = 1;

		double dp_mat[200] = { 0 };

		double max_score(0);
		double replace(0);

		if (row1 > 0 && col1 > 0)
		{
			for (int row = 0; row < row1; row++)
			{
				if (row > 0)
				{
					for (int col = 0; col < col1; col++)
					{
						if (col > 0)
						{
							double temp = replace + dp_node_value[row * nb_col + col];
							replace = dp_mat[col];

							if (dp_mat[col - 1] > dp_mat[col])
								dp_mat[col] = dp_mat[col - 1];

							if (temp > dp_mat[col])
								dp_mat[col] = temp;
						}
						else
						{
							replace = dp_mat[col];
							dp_mat[col] = dp_node_value[row * nb_col];
						}
					}
				}
				else
				{
					for (int col = 0; col < col1; col++)
						dp_mat[col] = dp_node_value[col];
				}
			}

			max_score += dp_mat[col1 - 1];
		}

		for (int row = 0; row < nb_row - row1; row++)
		{
			if (row > 0)
			{
				for (int col = 0; col < nb_col - col1; col++)
				{
					if (col > 0)
					{
						double temp = replace + dp_node_value[(row1 + row) * nb_col + col1 + col];
						replace = dp_mat[col];

						if (dp_mat[col - 1] > dp_mat[col])
							dp_mat[col] = dp_mat[col - 1];

						if (temp > dp_mat[col])
							dp_mat[col] = temp;
					}
					else
					{
						replace = dp_mat[col];
						dp_mat[col] = dp_node_value[(row1 + row) * nb_col + col1];
					}
				}
			}
			else
			{
				for (int col = 0; col < nb_col - col1; col++)
					dp_mat[col] = dp_node_value[row1 * nb_col + col1 + col];
			}
		}

		max_score += dp_mat[nb_col - col1 - 1];

		if (max_score < dali_score)
			gpu_domain[row1 * nb_col + col1] = 0;
	}
}

int ** algorithm_engineering(int nb_row, int nb_col, double dali_score, double * gpu_align_edge)
{
	int * cpu_domain = new int[nb_row * nb_col];
	int * gpu_domain;
	cudaMalloc((void**)&gpu_domain, sizeof(int) * nb_row * nb_col);

	int ** domain = new int *[nb_row];
	for (int i = 0; i < nb_row; i++)
		domain[i] = new int[nb_col];

	double * dp_node_value;
	cudaMalloc((void**)&dp_node_value, sizeof(double) * nb_row * nb_col);

	local_dp_1 << <nb_row * nb_col / 64 + 1, 64 >> >(nb_row, nb_col, gpu_align_edge, dp_node_value);
	local_dp_2 << <nb_row * nb_col / 64 + 1, 64 >> >(nb_row, nb_col, dp_node_value, gpu_domain, dali_score);

	cudaMemcpy(cpu_domain, gpu_domain, sizeof(int) * nb_row * nb_col, cudaMemcpyDeviceToHost);

	for (int i = 0; i < nb_row; i++)
	{
		for (int j = 0; j < nb_col; j++)
		{
			domain[i][j] = cpu_domain[i * nb_col + j];
		}
	}

	cudaFree(gpu_domain);
	cudaFree(dp_node_value);
	delete[] cpu_domain;

	return domain;
}