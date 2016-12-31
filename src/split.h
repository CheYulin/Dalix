#include <cmath>

#define MAX(x, y) ((x<y)?y:x)
#define MIN(x, y) ((x<y)?x:y)

class problem {
public:
    double ub;
    double lb;
    int up[120];
    int lo[120];

    problem(double ub1, double lb1, int *up1, int *lo1, int nb_col1) {
        ub = ub1;
        lb = lb1;
        for (int col = 0; col < nb_col1; col++) {
            up[col] = up1[col];
            lo[col] = lo1[col];
        }
    };

    ~problem() {};
};

void split(int nb_row, int nb_col, problem current_problem, int *lo1, int *up1, int *lo2, int *up2) {
    //int left;
    int best_col(-1), best_row(-1), min_area(0), max_area(0);

    int black[150][150];
    int white[150][150];

    /*****************************************
    * Step 1 - find the best splitting point *
    *****************************************/

    //Step 1.1 - Filling black values
    for (int row(nb_row - 1); row >= 0; --row) {
        if (row > current_problem.up[0]) {
            black[0][row] = 0;
        } else {
            black[0][row] = current_problem.up[0] - row;
        }
    }
    for (int col(1); col != nb_col; ++col) {
        for (int row(nb_row - 1); row >= 0; --row) {
            if (row > current_problem.up[col]) {
                black[col][row] = 0;
            } else {
                black[col][row] = (current_problem.up[col] - row) + (black[col - 1][row] + 1);
            }
        }
    }

    //Step 1.2 - filling white
    for (int row(0); row < nb_row; ++row) {
        if (row < current_problem.lo[nb_col - 1]) {
            white[nb_col - 1][row] = 0;
        } else {
            white[nb_col - 1][row] = row - current_problem.lo[nb_col - 1] + 1;
        }
    }
    for (int col(nb_col - 2); col >= 0; --col) {
        for (int row(0); row < nb_row; ++row) {
            if (row < current_problem.lo[col]) {
                white[col][row] = 0;
            } else {
                white[col][row] = (row - current_problem.lo[col] + 1) + white[col + 1][row];
            }
        }
    }

    for (int col(0); col != nb_col; ++col) {
        for (int row(current_problem.lo[col] + 1); row < current_problem.up[col]; row++) {
            if (MIN(black[col][row], white[col][row]) > min_area) {
                best_col = col;
                best_row = row;
                min_area = MIN(black[col][row], white[col][row]);
                max_area = MAX(black[col][row], white[col][row]);
            } else if (MIN(black[col][row], white[col][row]) == min_area) {
                if (MAX(black[col][row], white[col][row]) > max_area) {
                    best_col = col;
                    best_row = row;
                    min_area = MIN(black[col][row], white[col][row]);
                    max_area = MAX(black[col][row], white[col][row]);
                }
            }
        }
    }

    if (best_col == -1 || best_row == -1 || min_area == 0)
        return;
    //The problem cannot be anymore

    /*****************************************************************
    * Step 2 - Computing the borders of white and black sub-problems *
    *****************************************************************/

    //Step 2.1 - Black sub-problem
    for (int col(0); col != nb_col; ++col) {
        lo2[col] = current_problem.lo[col];

        if (col < best_col && current_problem.up[col] > best_row - 1)
            up2[col] = best_row - 1;
        else if (col == best_col && current_problem.up[col] > best_row)
            up2[col] = best_row;
        else
            up2[col] = current_problem.up[col];
    }

    //Step 2.2 - White sub-problem
    for (int col(0); col != nb_col; ++col) {
        up1[col] = current_problem.up[col];

        if (col >= best_col && current_problem.lo[col] < best_row)
            lo1[col] = best_row;
        else
            lo1[col] = current_problem.lo[col];
    }
}