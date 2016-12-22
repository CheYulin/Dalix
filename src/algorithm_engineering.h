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

int **algorithm_engineering(double ****align_edge, double **row_mat, double **col_mat, int nb_row, int nb_col,
                            double dali_score) {
    //time
    //struct timeb startTime, endTime;
    //ftime(&startTime);
    //time

    int **domain = new int *[nb_row];
    int nb_delete(0);

    for (int row = 0; row < nb_row; row++) {
        domain[row] = new int[nb_col];
        for (int col = 0; col < nb_col; col++) {
            domain[row][col] = 1;
        }
    }

    double **align_node = new double *[nb_row];
    for (int row = 0; row < nb_row; row++) {
        align_node[row] = new double[nb_col];
        for (int col = 0; col < nb_col; col++) {
            align_node[row][col] = 0;
        }
    }

    double dp_mat[150][150] = {0};
    //--------------------------------------------local dp
#pragma omp parallel for schedule(dynamic) private(dp_mat)
    for (int row1 = 0; row1 < nb_row; row1++) {
        for (int col1 = 0; col1 < nb_col; col1++) {
            for (int i = 0; i < nb_row - row1; i++)
                dp_mat[i][0] = 0;
            for (int i = 0; i < nb_col - col1; i++)
                dp_mat[0][i] = 0;
            for (int row2 = 1; row2 < nb_row - row1; row2++) {
                for (int col2 = 1; col2 < nb_col - col1; col2++) {
                    dp_mat[row2][col2] = dp_mat[row2][col2 - 1];

                    if (dp_mat[row2 - 1][col2] > dp_mat[row2][col2])
                        dp_mat[row2][col2] = dp_mat[row2 - 1][col2];
                    double temp = dp_mat[row2 - 1][col2 - 1] + align_edge[row1][col1][row2 - 1][col2 - 1];
                    if (temp > dp_mat[row2][col2])
                        dp_mat[row2][col2] = temp;
                }
            }
            align_node[row1][col1] = dp_mat[nb_row - row1 - 1][nb_col - col1 - 1] + 0.2;
        }
    }
    //--------------------------------------------local dp
#pragma omp parallel for schedule(dynamic) private(dp_mat)
    for (int row1 = 0; row1 < nb_row; row1++) {
        for (int col1 = 0; col1 < nb_col; col1++) {
            double max_score(0);
            //--------------------------------------------part 1
            if (row1 > 0 && col1 > 0) {
                for (int i = 0; i < row1; i++) {
                    dp_mat[i][0] = align_node[i][0];
                }
                for (int i = 0; i < col1; i++)
                    dp_mat[0][i] = align_node[0][i];

                for (int row = 1; row < row1; row++) {
                    for (int col = 1; col < col1; col++) {
                        dp_mat[row][col] = dp_mat[row][col - 1];

                        if (dp_mat[row - 1][col] > dp_mat[row][col])
                            dp_mat[row][col] = dp_mat[row - 1][col];

                        double temp = dp_mat[row - 1][col - 1] + align_node[row][col];
                        if (temp > dp_mat[row][col])
                            dp_mat[row][col] = temp;
                    }
                }

                max_score += dp_mat[row1 - 1][col1 - 1];
            }
            //--------------------------------------------part 1

            //--------------------------------------------part 2
            for (int i = 0; i < nb_row - row1; i++)
                dp_mat[i][0] = align_node[row1 + i][col1];

            for (int i = 0; i < nb_col - col1; i++)
                dp_mat[0][i] = align_node[row1][col1 + i];

            for (int row = 1; row < nb_row - row1; row++) {
                for (int col = 1; col < nb_col - col1; col++) {
                    dp_mat[row][col] = dp_mat[row][col - 1];

                    if (dp_mat[row - 1][col] > dp_mat[row][col])
                        dp_mat[row][col] = dp_mat[row - 1][col];

                    double temp = dp_mat[row - 1][col - 1] + align_node[row1 + row][col1 + col];
                    if (temp > dp_mat[row][col])
                        dp_mat[row][col] = temp;
                }
            }

            max_score += dp_mat[nb_row - row1 - 1][nb_col - col1 - 1];

            if (max_score < dali_score) {
                domain[row1][col1] = 0;
                nb_delete++;
            }
            //--------------------------------------------part 2
        }
    }

    for (int row1 = 0; row1 < nb_row; row1++)
        delete[] align_node[row1];
    delete[] align_node;

    //cout << "number of nodes deleted: " << nb_delete << endl;
    //ftime(&endTime);
    //cout << "algorithm engineering: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << endl;
    return domain;
}