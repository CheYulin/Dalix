#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <math.h>
#include <list>
#include "distance_matrix.h"
#include "algorithm_engineering.h"
#include "edge_dali.h"
#include "LR.h"

using namespace std;

int nb_row;
int nb_col;
int nb_branch_node;
int nb_solved_node(0);
double ** row_mat;
double ** col_mat;
double **** align_edge;
double global_ub(INFINITY);
double global_lb(-INFINITY);
int * alignment = new int[200];
string filename1;
string filename2;

list<problem> problem_list;

void insert_node(problem);
void delete_node();
void end();

int main()
{
	struct timeb startTime, endTime;

	int nb_iteration;
	int precise;
	double dali_score;
	double z_score;

	cout << "nb_B&B_node:	";
	cin >> nb_branch_node;
	cout << "nb_iteration:	";
	cin >> nb_iteration;
	cout << "protein  1:	";
	cin >> filename1;
	cout << "protein  2:	";
	cin >> filename2;
	cout << "Z score:	";
	cin >> z_score;
	cout << "precise:	";
	cin >> precise;

	ftime(&startTime);

	row_mat = distanceMatrix(filename1, &nb_row);
	col_mat = distanceMatrix(filename2, &nb_col);

	double L = sqrt((double)nb_row * (double)nb_col);
	double x = (L > 400.0) ? 400.0 : L;
	double m_L = 7.9494 + 0.70852*x + 2.5895*0.0001*x*x - 1.9156*0.000001*x*x*x;
	if (L > 400.0)
	{
		m_L = m_L + (L - 400.0)*1.0;
	}
	dali_score = z_score * 0.5 * m_L + m_L;
	cout << "dali score:	" << dali_score << endl;

	int ** domain;
	int align_edge_size = align_edge_size_compute(nb_row, nb_col);
	double * gpu_align_edge;
	cudaMalloc((void**)&gpu_align_edge, sizeof(double) * align_edge_size);

	double * gpu_row_mat;
	double * gpu_col_mat;
	cudaMalloc((void**)&gpu_row_mat, sizeof(double) * nb_row * nb_row);
	cudaMalloc((void**)&gpu_col_mat, sizeof(double) * nb_col * nb_col);
	double * cpu_row_mat = new double[nb_row * nb_row];
	double * cpu_col_mat = new double[nb_col * nb_col];
	for (int i = 0; i < nb_row; i++)
	{
		for (int j = 0; j < nb_row; j++)
			cpu_row_mat[i * nb_row + j] = row_mat[i][j];
	}
	for (int i = 0; i < nb_col; i++)
	{
		for (int j = 0; j < nb_col; j++)
			cpu_col_mat[i * nb_col + j] = col_mat[i][j];
	}
	cudaMemcpy(gpu_row_mat, cpu_row_mat, sizeof(double) * nb_row * nb_row, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_col_mat, cpu_col_mat, sizeof(double) * nb_col * nb_col, cudaMemcpyHostToDevice);

	gpu_edge_dalix << <nb_row * nb_col / 64 + 1, 64 >> >(gpu_align_edge, gpu_row_mat, gpu_col_mat, nb_row, nb_col);

	double * cpu_align_edge = new double[align_edge_size];
	cudaMemcpy(cpu_align_edge, gpu_align_edge, sizeof(double) * align_edge_size, cudaMemcpyDeviceToHost);

	domain = algorithm_engineering(nb_row, nb_col, dali_score, gpu_align_edge); //algorithm engineering

	int * up = new int[nb_col];
	int * lo = new int[nb_col];
	for (int col = 0; col < nb_col; col++)
	{
		lo[col] = -1;
		up[col] = nb_row;
		for (int row = nb_row - 1; domain[row][col] == 0 && row >= 0; row--)
			up[col] = row;
		for (int row = 0; domain[row][col] == 0 && row < nb_row; row++)
			lo[col] = row;
	}

	int *** lambda_for_cik = new int **[nb_row];
	for (int row = 0; row < nb_row; row++)
	{
		lambda_for_cik[row] = new int *[nb_col];
		for (int col = 0; col < nb_col; col++)
		{
			if (row < up[col] && row > lo[col])
			{
				lambda_for_cik[row][col] = new int[(nb_row - row - 1)*(nb_col - col - 1) * 2 + 1];
				int i = 1;
				int nb_lessthan0 = 0;
				for (int row2 = 0; row2 < nb_row - row - 1; row2++)
				{
					for (int col2 = 0; col2 < nb_col - col - 1; col2++)
					{
						if (cpu_align_edge[align_edge_iter_compute(nb_row, nb_col, row, col, row2, col2)] <= 0)
						{
							lambda_for_cik[row][col][i++] = row + row2 + 1;
							lambda_for_cik[row][col][i++] = col + col2 + 1;
							nb_lessthan0++;
						}
					}
				}
				lambda_for_cik[row][col][0] = nb_lessthan0;
			}
		}
	}

	int cpu_lambda_for_cik_size(0);
	for (int row = 0; row < nb_row; row++)
	{
		for (int col = 0; col < nb_col; col++)
		{
			if (row < up[col] && row > lo[col])
			{
				cpu_lambda_for_cik_size += lambda_for_cik[row][col][0] * 2;
			}
		}
	}

	int * cpu_lambda_for_cik = new int[cpu_lambda_for_cik_size + nb_row * nb_col];
	int temp_iter(nb_row * nb_col);
	for (int row = 0; row < nb_row; row++)
	{
		for (int col = 0; col < nb_col; col++)
		{
			if (row < up[col] && row > lo[col])
			{
				cpu_lambda_for_cik[row * nb_col + col] = temp_iter;
				temp_iter += lambda_for_cik[row][col][0] * 2;
				for (int i = 0; i < lambda_for_cik[row][col][0] * 2; i++)
				{
					cpu_lambda_for_cik[cpu_lambda_for_cik[row * nb_col + col] + i] = lambda_for_cik[row][col][i + 1];
				}
			}
		}
	}

	problem root(INFINITY, -INFINITY, up, lo, nb_col);

	ftime(&endTime);
	cout << "Allocation time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << endl;
	ftime(&startTime);

	//-------------------------------------------------------------------------------root problem
	if (solve_lr(nb_row, nb_col, gpu_align_edge, cpu_lambda_for_cik, &root, cpu_lambda_for_cik_size, nb_iteration, alignment, precise, global_lb) == 0)
	{
		global_ub = root.ub;
		global_lb = root.lb;
		problem_list.push_front(root);
	}
	else
	{
		ftime(&endTime);
		cout << "Solve time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << endl;
		end();
		return 0;
	}
	//-------------------------------------------------------------------------------root problem

	//-------------------------------------------------------------------------------B & B
	int * lo1 = new int[nb_col];
	int * up1 = new int[nb_col];
	int * lo2 = new int[nb_col];
	int * up2 = new int[nb_col];

	while (!problem_list.empty() && nb_branch_node != 0)
	{
		problem current_problem(problem_list.front());
		problem_list.pop_front();
		split(nb_row, nb_col, current_problem, lo1, up1, lo2, up2);

		problem sub_problema(INFINITY, -INFINITY, up1, lo1, nb_col);
		problem sub_problemb(INFINITY, -INFINITY, up2, lo2, nb_col);
		split(nb_row, nb_col, sub_problema, lo1, up1, lo2, up2);
		problem sub_problem1(INFINITY, -INFINITY, up1, lo1, nb_col);
		problem sub_problem2(INFINITY, -INFINITY, up2, lo2, nb_col);
		split(nb_row, nb_col, sub_problemb, lo1, up1, lo2, up2);
		problem sub_problem3(INFINITY, -INFINITY, up1, lo1, nb_col);
		problem sub_problem4(INFINITY, -INFINITY, up2, lo2, nb_col);

		int return_value;

		return_value = solve_lr(nb_row, nb_col, gpu_align_edge, cpu_lambda_for_cik, &sub_problem1, cpu_lambda_for_cik_size, nb_iteration, alignment, precise, global_lb);
		if (return_value == 0)
		{
			if (sub_problem1.lb > global_lb)
			{
				global_lb = sub_problem1.lb;
				global_ub = sub_problem1.ub;
			}
			insert_node(sub_problem1);
		}
		else if (return_value == 2)
		{
			cout << "Useless node." << endl;
		}
		else
		{
			ftime(&endTime);
			cout << "Solve time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << endl;
			end();
			return 0;
		}

		nb_solved_node += 1;
		if (nb_solved_node >= nb_branch_node)
		{
			ftime(&endTime);
			cout << "Solve time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << endl;
			end();
			return 0;
		}

		return_value = solve_lr(nb_row, nb_col, gpu_align_edge, cpu_lambda_for_cik, &sub_problem2, cpu_lambda_for_cik_size, nb_iteration, alignment, precise, global_lb);
		if (return_value == 0)
		{
			if (sub_problem2.lb > global_lb)
			{
				global_lb = sub_problem2.lb;
				global_ub = sub_problem2.ub;
			}
			insert_node(sub_problem2);
		}
		else if (return_value == 2)
		{
			cout << "Useless node." << endl;
		}
		else
		{
			ftime(&endTime);
			cout << "Solve time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << endl;
			end();
			return 0;
		}

		nb_solved_node += 1;
		if (nb_solved_node >= nb_branch_node)
		{
			ftime(&endTime);
			cout << "Solve time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << endl;
			end();
			return 0;
		}

		return_value = solve_lr(nb_row, nb_col, gpu_align_edge, cpu_lambda_for_cik, &sub_problem3, cpu_lambda_for_cik_size, nb_iteration, alignment, precise, global_lb);
		if (return_value == 0)
		{
			if (sub_problem3.lb > global_lb)
			{
				global_lb = sub_problem3.lb;
				global_ub = sub_problem3.ub;
			}
			insert_node(sub_problem3);
		}
		else if (return_value == 2)
		{
			cout << "Useless node." << endl;
		}
		else
		{
			ftime(&endTime);
			cout << "Solve time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << endl;
			end();
			return 0;
		}

		nb_solved_node += 1;
		if (nb_solved_node >= nb_branch_node)
		{
			ftime(&endTime);
			cout << "Solve time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << endl;
			end();
			return 0;
		}

		return_value = solve_lr(nb_row, nb_col, gpu_align_edge, cpu_lambda_for_cik, &sub_problem4, cpu_lambda_for_cik_size, nb_iteration, alignment, precise, global_lb);
		if (return_value == 0)
		{
			if (sub_problem4.lb > global_lb)
			{
				global_lb = sub_problem4.lb;
				global_ub = sub_problem4.ub;
			}
			insert_node(sub_problem4);
		}
		else if (return_value == 2)
		{
			cout << "Useless node." << endl;
		}
		else
		{
			ftime(&endTime);
			cout << "Solve time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << endl;
			end();
			return 0;
		}

		nb_solved_node += 1;
		if (nb_solved_node >= nb_branch_node)
		{
			ftime(&endTime);
			cout << "Solve time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << endl;
			end();
			return 0;
		}

		delete_node();

		while (problem_list.size() > 20)
		{
			problem_list.pop_back();
		}
	}

	delete[] lo1, up1, lo2, up2;

	for (int row = 0; row < nb_row; row++)
	{
		for (int col = 0; col < nb_col; col++)
		{
			if (row < up[col] && row > lo[col])
			{
				delete[] lambda_for_cik[row][col];
			}
		}
		delete[] lambda_for_cik[row];
	}
	delete[] lambda_for_cik;
	delete[] up, lo;

	for (int row = 0; row < nb_row; row++)
	{
		delete[] domain[row];
	}
	delete[] domain;
	//-------------------------------------------------------------------------------B & B

	ftime(&endTime);
	cout << "Solve time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << endl;
	end();
	return 0;
}

void insert_node(problem current_problem)
{
	if (current_problem.ub >= global_lb)
	{
		int inserted(0);
		double new_lb(current_problem.lb);

		if (problem_list.empty())
			problem_list.push_front(current_problem);
		else
		{
			list<problem>::iterator it_list = problem_list.begin();
			while (it_list != problem_list.end())
			{
				double current_lb((*it_list).lb);

				if (new_lb >= current_lb)
				{
					problem_list.insert(it_list, current_problem);
					it_list = problem_list.end();
					inserted = 1;
				}
				else
					++it_list;
			}
			if (inserted == 0)
				problem_list.push_back(current_problem);
		}
	}
}

void delete_node()
{
	list<problem>::iterator it_list;
	for (it_list = problem_list.begin(); it_list != problem_list.end();)
	{
		double current_ub((*it_list).ub);
		if (current_ub <= global_lb)
		{
			it_list = problem_list.erase(it_list);
		}
		else
		{
			it_list++;
		}
	}
}

void end()
{
	cout << "global_ub: " << global_ub << endl;
	cout << "global_lb: " << global_lb << endl;
	cout << "B&B node: " << nb_solved_node << endl;
	cout << filename1 << ":	" << filename2 << endl;
	for (int i = 0; i < nb_row; i++)
	{
		cout << i << ":		" << alignment[i] << endl;
	}
	delete[] alignment;
}
