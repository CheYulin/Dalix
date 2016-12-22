#include <iostream>
#include <math.h>
#include <sys/timeb.h>
#include "split.h"
#include "omp.h"

using namespace std;

static omp_lock_t lock;

int solve_lr(int nb_row, int nb_col, double ****align_edge, int ***lambda_for_cik, problem *current_problem) {
    omp_init_lock(&lock);

    double global_ub(INFINITY);
    double global_lb(-INFINITY);

    double ***lambda_h = new double **[nb_row];
    double ***lambda_v = new double **[nb_row];
    double ***lambda_a = new double **[nb_row];
    double ***g_h = new double **[nb_row];
    double ***g_v = new double **[nb_row];
    double ***g_a = new double **[nb_row];

    for (int i = 0; i < nb_row; i++) {
        lambda_h[i] = new double *[nb_col];
        for (int j = 0; j < nb_col; j++) {
            if (i > 0) {
                lambda_h[i][j] = new double[i];
                for (int z = 0; z < i; z++)
                    lambda_h[i][j][z] = 0;
            } else
                lambda_h[i][j] = NULL;
        }
    }
    for (int i = 0; i < nb_row; i++) {
        lambda_v[i] = new double *[nb_col];
        for (int j = 0; j < nb_col; j++) {
            if (j > 0) {
                lambda_v[i][j] = new double[j];
                for (int z = 0; z < j; z++)
                    lambda_v[i][j][z] = 0;
            } else
                lambda_v[i][j] = NULL;
        }
    }
    for (int i = 0; i < nb_row; i++) {
        lambda_a[i] = new double *[nb_col];
        for (int j = 0; j < nb_col; j++) {
            if (i > 0) {
                lambda_a[i][j] = new double[i];
                for (int z = 0; z < i; z++)
                    lambda_a[i][j][z] = 0;
            } else
                lambda_a[i][j] = NULL;
        }
    }
    for (int i = 0; i < nb_row; i++) {
        g_h[i] = new double *[nb_col];
        for (int j = 0; j < nb_col; j++) {
            if (i > 0) {
                g_h[i][j] = new double[i];
                for (int z = 0; z < i; z++)
                    g_h[i][j][z] = 0;
            } else
                g_h[i][j] = NULL;
        }
    }
    for (int i = 0; i < nb_row; i++) {
        g_v[i] = new double *[nb_col];
        for (int j = 0; j < nb_col; j++) {
            if (j > 0) {
                g_v[i][j] = new double[j];
                for (int z = 0; z < j; z++)
                    g_v[i][j][z] = 0;
            } else
                g_v[i][j] = NULL;
        }
    }
    for (int i = 0; i < nb_row; i++) {
        g_a[i] = new double *[nb_col];
        for (int j = 0; j < nb_col; j++) {
            if (i > 0) {
                g_a[i][j] = new double[i];
                for (int z = 0; z < i; z++)
                    g_a[i][j][z] = 0;
            } else
                g_a[i][j] = NULL;
        }
    }

    double alpha(1);
    int nb_improve(0);
    int nb_noimprove(0);
    double last_ub(0);
    double last_lb(0);
    double dp_mat[100][100] = {0};
    int path_mat[100][100] = {0};
    double align_node[100][100] = {0};
    bool need_to_change[100][100] = {0};
    memset(need_to_change, 1, 100 * 100 * sizeof(bool));

    for (int iteration = 0; iteration < 500; iteration++) {
        double ub(0);
        double lb(0);
        //---------------------------------------------------------------------double dp
        bool **align_node_xvalue = new bool *[nb_row];
        for (int row = 0; row < nb_row; row++) {
            align_node_xvalue[row] = new bool[nb_col];
            for (int col = 0; col < nb_col; col++) {
                align_node_xvalue[row][col] = 0;
            }
        }
        //--------------------------------------------local dp
#pragma omp parallel for schedule(dynamic)
        for (int row1 = 0; row1 < nb_row; row1++) {
            double align_node_temp[100] = {0};
            double dp_mat_test[100] = {0};
            for (int col1 = 0; col1 < nb_col; col1++) {
                if (row1 < current_problem->up[col1] && row1 > current_problem->lo[col1]) {
                    if (need_to_change[row1][col1]) {
                        double replace(0);
                        for (int row2 = 0; row2 < nb_row - row1; row2++) {
                            if (row2 != 0) {
                                for (int col2 = 0; col2 < nb_col - col1; col2++) {
                                    if (col2 != 0) {
                                        double temp;
                                        double temp2 = align_edge[row1][col1][row2 - 1][col2 - 1];

                                        if (row1 + row2 < current_problem->up[col1 + col2] &&
                                            row1 + row2 > current_problem->lo[col1 + col2])
                                            temp = replace + temp2
                                                   - lambda_h[row1 + row2][col1 + col2][row1]
                                                   - lambda_v[row1 + row2][col1 + col2][col1]
                                                   + (temp2 > 0 ? 0 : lambda_a[row1 + row2][col1 + col2][row1]);
                                        else
                                            temp = -INFINITY;

                                        replace = dp_mat_test[col2];

                                        if (dp_mat_test[col2] < dp_mat_test[col2 - 1])
                                            dp_mat_test[col2] = dp_mat_test[col2 - 1];
                                        if (temp > dp_mat_test[col2])
                                            dp_mat_test[col2] = temp;
                                    } else {
                                        replace = dp_mat_test[0];
                                        if (row2 + row1 < current_problem->up[col1] &&
                                            row2 + row1 > current_problem->lo[col1])
                                            dp_mat_test[0] = 0;
                                        else
                                            dp_mat_test[0] = -INFINITY;
                                    }
                                }
                            } else {
                                for (int col2 = 0; col2 < nb_col - col1; col2++) {
                                    if (row1 < current_problem->up[col2 + col1] &&
                                        row1 > current_problem->lo[col1 + col2])
                                        dp_mat_test[col2] = 0;
                                    else
                                        dp_mat_test[col2] = -INFINITY;
                                }
                            }
                        }
                        align_node_temp[col1] = dp_mat_test[nb_col - col1 - 1] + 0.2;
                        for (int j = 0; j < row1; j++) {
                            align_node_temp[col1] += lambda_h[row1][col1][j] - lambda_a[row1][col1][j];
                        }
                        for (int l = 0; l < col1; l++) {
                            align_node_temp[col1] += lambda_v[row1][col1][l];
                        }
                        for (int i = 1; i < lambda_for_cik[row1][col1][0] * 2 + 1;) {
                            int r = lambda_for_cik[row1][col1][i++];
                            int s = lambda_for_cik[row1][col1][i++];
                            align_node_temp[col1] -= lambda_a[r][s][row1];
                        }
                    } else
                        align_node_temp[col1] = align_node[row1][col1];
                } else
                    align_node_temp[col1] = -INFINITY;
            }
            omp_set_lock(&lock);
            memcpy(align_node[row1], align_node_temp, nb_col * sizeof(double));
            omp_unset_lock(&lock);
        }

        memset(need_to_change, 0, 100 * 100 * sizeof(bool));
        //--------------------------------------------local dp
        //--------------------------------------------global dp
        for (int i = 0; i < nb_row; i++) {
            if (i < current_problem->up[0] && i > current_problem->lo[0]) {
                dp_mat[i][0] = align_node[i][0];
                path_mat[i][0] = -2;
            } else {
                dp_mat[i][0] = -INFINITY;
                path_mat[i][0] = -2;
            }
        }
        for (int i = 0; i < nb_col; i++) {
            if (0 < current_problem->up[i] && 0 > current_problem->lo[i]) {
                dp_mat[0][i] = align_node[0][i];
                path_mat[0][i] = -1;
            } else {
                dp_mat[0][i] = -INFINITY;
                path_mat[0][i] = -1;
            }
        }

        for (int row = 1; row < nb_row; row++) {
            for (int col = 1; col < nb_col; col++) {
                dp_mat[row][col] = dp_mat[row][col - 1];
                path_mat[row][col] = -1;

                double temp = dp_mat[row - 1][col];
                if (temp > dp_mat[row][col]) {
                    dp_mat[row][col] = temp;
                    path_mat[row][col] = -2;
                }

                temp = dp_mat[row - 1][col - 1] + align_node[row][col];
                if (temp > dp_mat[row][col]) {
                    dp_mat[row][col] = temp;
                    path_mat[row][col] = 1;
                }
            }
        }

        ub = dp_mat[nb_row - 1][nb_col - 1];  //upper bound
        //---------------------------------------------flash back
        int x_1 = nb_row;
        int y_1 = nb_col;
        while (x_1 != 1 || y_1 != 1) {
            if (path_mat[x_1 - 1][y_1 - 1] == -1)
                y_1--;
            else if (path_mat[x_1 - 1][y_1 - 1] == -2)
                x_1--;
            else if (path_mat[x_1 - 1][y_1 - 1] == 1) {
                align_node_xvalue[x_1 - 1][y_1 - 1] = true;
                x_1--;
                y_1--;
                if (x_1 - 1 == 0 || y_1 - 1 == 0)
                    align_node_xvalue[x_1 - 1][y_1 - 1] = 1;
            }
        }
        //---------------------------------------------flash back
        //--------------------------------------------global dp
        //--------------------------------------------compute lower bound
#pragma omp parallel for schedule(dynamic) reduction(+: lb)
        for (int row1 = 0; row1 < nb_row; row1++) {
            for (int col1 = 0; col1 < nb_col; col1++) {
                if (align_node_xvalue[row1][col1]) {
                    lb += 0.2;
                    for (int row2 = 0; row2 < nb_row - row1 - 1; row2++) {
                        for (int col2 = 0; col2 < nb_col - col1 - 1; col2++) {
                            if (align_node_xvalue[row1 + row2 + 1][col1 + col2 + 1])
                                lb += align_edge[row1][col1][row2][col2];
                        }
                    }
                }
            }
        }
        //--------------------------------------------compute lower bound
        if (ub < global_ub)
            global_ub = ub;
        if (lb > global_lb)
            global_lb = lb;

        if (iteration > 0) {
            if (ub < last_ub || lb > last_lb) {
                nb_improve++;
                nb_noimprove = 0;
                if (nb_improve >= 5)
                    alpha /= 0.9;
            } else {
                nb_improve = 0;
                nb_noimprove++;
                if (nb_noimprove >= 5)
                    alpha *= 0.9;
            }
        }

        last_ub = ub;
        last_lb = lb;

        if (global_ub <= global_lb) {
            cout << "iteration: " << iteration << endl;
            cout << "final_solution: " << global_lb << endl;
            return 0;
        }
        //---------------------------------------------------------------------double dp
        //---------------------------------------------------------------------lambda update
        double thelta(0);
        //---------------------------------------------------------------------update g
#pragma omp parallel for schedule(dynamic)
        for (int row1 = 0; row1 < nb_row; row1++) {
            for (int col1 = 0; col1 < nb_col; col1++) {
                if (row1 < current_problem->up[col1] && row1 > current_problem->lo[col1]) {
                    for (int j_1 = 0; j_1 < row1; j_1++) {
                        g_a[row1][col1][j_1] += 1.0;
                    }
                }
                if (align_node_xvalue[row1][col1]) {
                    for (int j = 0; j < row1; j++) {
                        g_h[row1][col1][j] += 1.0;
                        g_a[row1][col1][j] -= 1.0;
                    }
                    for (int l = 0; l < col1; l++) {
                        g_v[row1][col1][l] += 1.0;
                    }
                }
            }
        }

#pragma omp parallel for schedule(dynamic)
        for (int row1 = 0; row1 < nb_row; row1++) {
            for (int col1 = 0; col1 < nb_col; col1++) {
                if (align_node_xvalue[row1][col1]) {
                    for (int i = 1; i < lambda_for_cik[row1][col1][0] * 2 + 1;) {
                        int r = lambda_for_cik[row1][col1][i++];
                        int s = lambda_for_cik[row1][col1][i++];
                        g_a[r][s][row1] -= 1.0;
                    }
                }
            }
        }

/*#pragma omp parallel for schedule(dynamic) private(dp_mat, path_mat)
		for (int row1 = 0; row1 < nb_row; row1++)
		{
			for (int col1 = 0; col1 < nb_col; col1++)
			{
				if (align_node_xvalue[row1][col1])
				{
					for (int x = -2; x < 3; x++)
					{
						if (row1 + x >= 0 && row1 + x < nb_row)
							need_to_change[row1 + x][col1] = true;
					}
					for (int i = 0; i < nb_row - row1; i++)
					{
						if (i + row1 < current_problem->up[col1] && i + row1 > current_problem->lo[col1])
						{
							dp_mat[i][0] = 0; path_mat[i][0] = -2;
						}
						else
						{
							dp_mat[i][0] = -INFINITY; path_mat[i][0] = -2;
						}
					}
					for (int i = 0; i < nb_col - col1; i++)
					{
						if (row1 < current_problem->up[i + col1] && row1 > current_problem->lo[i + col1])
						{
							dp_mat[0][i] = 0; path_mat[0][i] = -1;
						}
						else
						{
							dp_mat[0][i] = -INFINITY; path_mat[0][i] = -1;
						}
					}

					for (int row2 = 1; row2 < nb_row - row1; row2++)
					{
						for (int col2 = 1; col2 < nb_col - col1; col2++)
						{
							dp_mat[row2][col2] = dp_mat[row2][col2 - 1];
							path_mat[row2][col2] = -1;

							double temp = dp_mat[row2 - 1][col2];
							if (temp > dp_mat[row2][col2])
							{
								dp_mat[row2][col2] = temp;
								path_mat[row2][col2] = -2;
							}

							double temp2 = align_edge[row1][col1][row2 - 1][col2 - 1];
							if (row1 + row2 < current_problem->up[col1 + col2] && row1 + row2 > current_problem->lo[col1 + col2])
								temp = dp_mat[row2 - 1][col2 - 1] + temp2
								- lambda_h[row1 + row2][col1 + col2][row1]
								- lambda_v[row1 + row2][col1 + col2][col1]
								+ (temp2 <= 0 ? lambda_a[row1 + row2][col1 + col2][row1] : 0);
							else
								temp = -INFINITY;
							if (temp > dp_mat[row2][col2])
							{
								dp_mat[row2][col2] = temp;
								path_mat[row2][col2] = 1;
							}
						}
					}

					int x = nb_row - row1;
					int y = nb_col - col1;
					while (x != 1 || y != 1)
					{
						if (path_mat[x - 1][y - 1] == -1)
							y--;
						else if (path_mat[x - 1][y - 1] == -2)
							x--;
						else if (path_mat[x - 1][y - 1] == 1)
						{
							g_h[row1 + x - 1][col1 + y - 1][row1] -= 1.0;
							g_v[row1 + x - 1][col1 + y - 1][col1] -= 1.0;
							if (align_edge[row1][col1][x - 2][y - 2] <= 0)
								g_a[row1 + x - 1][col1 + y - 1][row1] += 1.0;
							x--; y--;
						}
					}
				}
			}
		}*/

#pragma omp parallel for schedule(dynamic) private(path_mat)
        for (int row1 = 0; row1 < nb_row; row1++) {
            double dp_mat_test[100] = {0};
            for (int col1 = 0; col1 < nb_col; col1++) {
                if (align_node_xvalue[row1][col1]) {
                    for (int x = -2; x < 3; x++) {
                        if (row1 + x >= 0 && row1 + x < nb_row)
                            need_to_change[row1 + x][col1] = true;
                    }
                    //---------------------------------------------------------------------y
                    double replace(0);
                    for (int row2 = 0; row2 < nb_row - row1; row2++) {
                        if (row2 != 0) {
                            for (int col2 = 0; col2 < nb_col - col1; col2++) {
                                if (col2 != 0) {
                                    double temp;
                                    double temp2 = align_edge[row1][col1][row2 - 1][col2 - 1];

                                    if (row1 + row2 < current_problem->up[col1 + col2] &&
                                        row1 + row2 > current_problem->lo[col1 + col2])
                                        temp = replace + temp2
                                               - lambda_h[row1 + row2][col1 + col2][row1]
                                               - lambda_v[row1 + row2][col1 + col2][col1]
                                               + (temp2 > 0 ? 0 : lambda_a[row1 + row2][col1 + col2][row1]);
                                    else
                                        temp = -INFINITY;

                                    replace = dp_mat_test[col2];

                                    if (dp_mat_test[col2] <= dp_mat_test[col2 - 1]) {
                                        dp_mat_test[col2] = dp_mat_test[col2 - 1];
                                        path_mat[row2][col2] = -1;
                                    } else {
                                        path_mat[row2][col2] = -2;
                                    }
                                    if (temp > dp_mat_test[col2]) {
                                        dp_mat_test[col2] = temp;
                                        path_mat[row2][col2] = 1;
                                    }
                                } else {
                                    replace = dp_mat_test[0];
                                    if (row2 + row1 < current_problem->up[col1] &&
                                        row2 + row1 > current_problem->lo[col1]) {
                                        dp_mat_test[0] = 0;
                                        path_mat[row2][0] = -2;
                                    } else {
                                        dp_mat_test[0] = -INFINITY;
                                        path_mat[row2][0] = -2;
                                    }
                                }
                            }
                        } else {
                            for (int col2 = 0; col2 < nb_col - col1; col2++) {
                                if (row1 < current_problem->up[col2 + col1] &&
                                    row1 > current_problem->lo[col1 + col2]) {
                                    dp_mat_test[col2] = 0;
                                    path_mat[0][col2] = -1;
                                } else {
                                    dp_mat_test[col2] = -INFINITY;
                                    path_mat[0][col2] = -1;
                                }
                            }
                        }
                    }
                    int x = nb_row - row1;
                    int y = nb_col - col1;
                    while (x != 1 || y != 1) {
                        if (path_mat[x - 1][y - 1] == -1)
                            y--;
                        else if (path_mat[x - 1][y - 1] == -2)
                            x--;
                        else if (path_mat[x - 1][y - 1] == 1) {
                            g_h[row1 + x - 1][col1 + y - 1][row1] -= 1.0;
                            g_v[row1 + x - 1][col1 + y - 1][col1] -= 1.0;
                            if (align_edge[row1][col1][x - 2][y - 2] <= 0)
                                g_a[row1 + x - 1][col1 + y - 1][row1] += 1.0;
                            x--;
                            y--;
                        }
                    }
                }
            }
        }
        //---------------------------------------------------------------------update g
        //---------------------------------------------------------------------compute thelta
        double temp(0);
#pragma omp parallel for schedule(dynamic) reduction(+: temp)
        for (int row1 = 0; row1 < nb_row; row1++) {
            for (int col1 = 0; col1 < nb_col; col1++) {
                if (row1 < current_problem->up[col1] && row1 > current_problem->lo[col1]) {
                    for (int j_1 = 0; j_1 < row1; j_1++) {
                        if (g_h[row1][col1][j_1] != 0)
                            temp++;
                        if (g_a[row1][col1][j_1] != 0)
                            temp++;
                    }
                    for (int l_1 = 0; l_1 < col1; l_1++) {
                        if (g_v[row1][col1][l_1] != 0)
                            temp++;
                    }
                }
            }
        }
        thelta = alpha * (ub - global_lb) / temp;
        //---------------------------------------------------------------------compute thelta
        //---------------------------------------------------------------------update lambda
#pragma omp parallel for schedule(dynamic)
        for (int row1 = 0; row1 < nb_row; row1++) {
            for (int col1 = 0; col1 < nb_col; col1++) {
                if (row1 < current_problem->up[col1] && row1 > current_problem->lo[col1]) {
                    for (int j = 0; j < row1; j++) {
                        lambda_h[row1][col1][j] -= thelta * g_h[row1][col1][j];
                        g_h[row1][col1][j] = 0;
                        if (lambda_h[row1][col1][j] < 0)
                            lambda_h[row1][col1][j] = 0;

                        lambda_a[row1][col1][j] -= thelta * g_a[row1][col1][j];
                        g_a[row1][col1][j] = 0;
                        if (lambda_a[row1][col1][j] < 0)
                            lambda_a[row1][col1][j] = 0;
                    }
                    for (int l = 0; l < col1; l++) {
                        lambda_v[row1][col1][l] -= thelta * g_v[row1][col1][l];
                        g_v[row1][col1][l] = 0;
                        if (lambda_v[row1][col1][l] < 0)
                            lambda_v[row1][col1][l] = 0;
                    }
                }
            }
        }
        //---------------------------------------------------------------------update lambda
        //---------------------------------------------------------------------lambda update
        for (int row1 = 0; row1 < nb_row; row1++) {
            delete[] align_node_xvalue[row1];
        }
        delete[] align_node_xvalue;
    }
    //---------------------------------------------------------------------LR
    for (int i = 0; i < nb_row; i++) {
        for (int j = 0; j < nb_col; j++) {
            delete[] lambda_h[i][j];
            delete[] lambda_v[i][j];
            delete[] lambda_a[i][j];
            delete[] g_h[i][j];
            delete[] g_v[i][j];
            delete[] g_a[i][j];
        }
        delete[] lambda_h[i];
        delete[] lambda_v[i];
        delete[] lambda_a[i];
        delete[] g_h[i];
        delete[] g_a[i];
        delete[] g_v[i];
    }
    delete[] g_h;
    delete[] g_a;
    delete[] g_v;
    delete[] lambda_h;
    delete[] lambda_v;
    delete[] lambda_a;

    current_problem->ub = global_ub;
    current_problem->lb = global_lb;

    return 1;
}