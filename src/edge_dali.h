#include <math.h>

double edge_dali_ae(double **row_mat, double **col_mat, int i, int k, int j, int l);

double ****edge_dalix(double **row_mat, double **col_mat, int nb_row, int nb_col) {
    double ****align_edge = new double ***[nb_row];
#pragma omp parallel for schedule(dynamic)
    for (int row1 = 0; row1 < nb_row; row1++) {
        align_edge[row1] = new double **[nb_col];
        for (int col1 = 0; col1 < nb_col; col1++) {
            align_edge[row1][col1] = new double *[nb_row - row1 - 1];
            for (int row2 = 0; row2 < nb_row - row1 - 1; row2++) {
                align_edge[row1][col1][row2] = new double[nb_col - col1 - 1];
                for (int col2 = 0; col2 < nb_col - col1 - 1; col2++) {
                    align_edge[row1][col1][row2][col2] = edge_dali_ae(row_mat, col_mat, row1, col1, row1 + row2 + 1,
                                                                      col1 + col2 + 1);
                }
            }
        }
    }
    return align_edge;
}

double edge_dali_ae(double **row_mat, double **col_mat, int i, int k, int j, int l) {
    double dA = row_mat[i][j];
    double dB = col_mat[k][l];
    if (dA != 0.0 && dB != 0.0) {
        return 2 * (0.2 - 2 * (fabs((double) dA - dB)) / (dA + dB)) *
               exp(-((dA + dB) * (dA + dB)) / (4.0 * 20.0 * 20.0));
    } else {
        return 0.0;
    }
}