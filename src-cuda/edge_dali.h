#include <math.h>

__device__ int gpu_align_edge_iter_compute1(int nb_row, int nb_col, int row1, int col1, int row2, int col2)
{
	return row1 * (2 * nb_row - row1 - 1) * nb_col * (nb_col - 1) / 4
		+ (nb_row - row1 - 1) * col1 * (2 * nb_col - col1 - 1) / 2
		+ row2 * (nb_col - col1 - 1) + col2;
}

__device__ double gpu_edge_dali_ae(int nb_row, int nb_col, double *row_mat, double *col_mat, int i, int k, int j, int l)
{
	double dA = row_mat[i * nb_row + j];
	double dB = col_mat[k * nb_col + l];
	if (dA != 0.0 && dB != 0.0)
	{
		return 2 * (0.2 - 2 * (fabs((double)dA - dB)) / (dA + dB))*exp(-((dA + dB)*(dA + dB)) / (4.0 * 20.0*20.0));
	}
	else
	{
		return 0.0;
	}
}

__global__ void gpu_edge_dalix(double * gpu_align_edge, double *gpu_row_mat, double *gpu_col_mat, int nb_row, int nb_col)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	if (bx * 64 + tx < nb_row * nb_col)
	{
		int row1 = (bx * 64 + tx) / nb_col;
		int col1 = (bx * 64 + tx) % nb_col;

		for (int row2 = 0; row2 < nb_row - row1 - 1; row2++)
		{
			for (int col2 = 0; col2 < nb_col - col1 - 1; col2++)
			{
				gpu_align_edge[gpu_align_edge_iter_compute1(nb_row, nb_col, row1, col1, row2, col2)] = gpu_edge_dali_ae(nb_row, nb_col, gpu_row_mat, gpu_col_mat, row1, col1, row1 + row2 + 1, col1 + col2 + 1);
			}
		}
	}
}

double edge_dali_ae(double **row_mat, double **col_mat, int i, int k, int j, int l)
{
	double dA = row_mat[i][j];
	double dB = col_mat[k][l];
	if (dA != 0.0 && dB != 0.0)
	{
		return 2 * (0.2 - 2 * (fabs((double)dA - dB)) / (dA + dB))*exp(-((dA + dB)*(dA + dB)) / (4.0 * 20.0*20.0));
	}
	else
	{
		return 0.0;
	}
}

double **** edge_dalix(double **row_mat, double **col_mat, int nb_row, int nb_col)
{
	double **** align_edge = new double ***[nb_row];
	for (int row1 = 0; row1 < nb_row; row1++)
	{
		align_edge[row1] = new double **[nb_col];
		for (int col1 = 0; col1 < nb_col; col1++)
		{
			align_edge[row1][col1] = new double *[nb_row - row1 - 1];
			for (int row2 = 0; row2 < nb_row - row1 - 1; row2++)
			{
				align_edge[row1][col1][row2] = new double[nb_col - col1 - 1];
				for (int col2 = 0; col2 < nb_col - col1 - 1; col2++)
				{
					align_edge[row1][col1][row2][col2] = edge_dali_ae(row_mat, col_mat, row1, col1, row1 + row2 + 1, col1 + col2 + 1);
				}
			}
		}
	}
	return align_edge;
}