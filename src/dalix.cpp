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
int nb_branch_node(4);
int nb_solved_node(1);
double **row_mat;
double **col_mat;
double ****align_edge;
double global_ub(INFINITY);
double global_lb(-INFINITY);

list <problem> problem_list;

void insert_node(problem);

void delete_node();

void end();

int main(int argc, char *argv[]) {
    struct timeb startTime, endTime;
    ftime(&startTime);

    string filename1 = argv[1];
    string filename2 = argv[2];

    row_mat = distanceMatrix(filename1, &nb_row);
    col_mat = distanceMatrix(filename2, &nb_col);

    int **domain = new int *[nb_row];
    for (int row = 0; row < nb_row; row++) {
        domain[row] = new int[nb_col];
        for (int col = 0; col < nb_col; col++)
            domain[row][col] = 1;
    }

    align_edge = edge_dalix(row_mat, col_mat, nb_row, nb_col);//compute dali score for each edge

    domain = algorithm_engineering(align_edge, row_mat, col_mat, nb_row, nb_col, 118.33); //algorithm engineering

    int *up = new int[nb_col];
    int *lo = new int[nb_col];
    for (int col = 0; col < nb_col; col++) {
        lo[col] = -1;
        up[col] = nb_row;
        for (int row = nb_row - 1; domain[row][col] == 0 && row >= 0; row--)
            up[col] = row;
        for (int row = 0; domain[row][col] == 0 && row < nb_row; row++)
            lo[col] = row;
    }

    int ***lambda_for_cik = new int **[nb_row];//compute all edges with dali score less than 0
#pragma omp parallel for schedule(dynamic)
    for (int row = 0; row < nb_row; row++) {
        lambda_for_cik[row] = new int *[nb_col];
        for (int col = 0; col < nb_col; col++) {
            if (row < up[col] && row > lo[col]) {
                lambda_for_cik[row][col] = new int[(nb_row - row - 1) * (nb_col - col - 1) * 2 + 1];
                int i = 1;
                int nb_lessthan0 = 0;
                for (int row2 = 0; row2 < nb_row - row - 1; row2++) {
                    for (int col2 = 0; col2 < nb_col - col - 1; col2++) {
                        if (align_edge[row][col][row2][col2] <= 0) {
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

    problem root(INFINITY, -INFINITY, up, lo, nb_col);

    for (int row = 0; row < nb_row; row++) {
        delete[] domain[row];
    }
    delete[] domain;

    ftime(&endTime);
    cout << "Allocation time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm)
         << endl;
    ftime(&startTime);

    //-------------------------------------------------------------------------------root problem
    if (solve_lr(nb_row, nb_col, align_edge, lambda_for_cik, &root) != 0) {
        global_ub = root.ub;
        global_lb = root.lb;
        problem_list.push_front(root);
    } else {
        ftime(&endTime);
        cout << "Solve time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm)
             << endl;
        system("pause");
        return 0;
    }
    //-------------------------------------------------------------------------------root problem

    //-------------------------------------------------------------------------------B & B
    int *lo1 = new int[nb_col];
    int *up1 = new int[nb_col];
    int *lo2 = new int[nb_col];
    int *up2 = new int[nb_col];

    while (nb_solved_node <= nb_branch_node && !problem_list.empty()) {
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

        if (solve_lr(nb_row, nb_col, align_edge, lambda_for_cik, &sub_problem1) != 0) {
            if (sub_problem1.lb > global_lb) {
                global_lb = sub_problem1.lb;
                global_ub = sub_problem1.ub;
            }
            insert_node(sub_problem1);
        } else {
            ftime(&endTime);
            cout << "Solve time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm)
                 << endl;
            end();
            return 0;
        }

        nb_solved_node += 1;

        if (solve_lr(nb_row, nb_col, align_edge, lambda_for_cik, &sub_problem2) != 0) {
            if (sub_problem2.lb > global_lb) {
                global_lb = sub_problem2.lb;
                global_ub = sub_problem2.ub;
            }
            insert_node(sub_problem2);
        } else {
            ftime(&endTime);
            cout << "Solve time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm)
                 << endl;
            end();
            return 0;
        }

        nb_solved_node += 1;

        if (solve_lr(nb_row, nb_col, align_edge, lambda_for_cik, &sub_problem3) != 0) {
            if (sub_problem3.lb > global_lb) {
                global_lb = sub_problem3.lb;
                global_ub = sub_problem3.ub;
            }
            insert_node(sub_problem3);
        } else {
            ftime(&endTime);
            cout << "Solve time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm)
                 << endl;
            end();
            return 0;
        }

        nb_solved_node += 1;

        if (solve_lr(nb_row, nb_col, align_edge, lambda_for_cik, &sub_problem4) != 0) {
            if (sub_problem4.lb > global_lb) {
                global_lb = sub_problem4.lb;
                global_ub = sub_problem4.ub;
            }
            insert_node(sub_problem4);
        } else {
            ftime(&endTime);
            cout << "Solve time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm)
                 << endl;
            end();
            return 0;
        }

        nb_solved_node += 1;

        delete_node();

        while (problem_list.size() > 20) {
            problem_list.pop_back();
        }
    }

    delete[] lo1, up1, lo2, up2;

    for (int row = 0; row < nb_row; row++) {
        for (int col = 0; col < nb_col; col++) {
            if (row < up[col] && row > lo[col]) {
                delete[] lambda_for_cik[row][col];
            }
        }
        delete[] lambda_for_cik[row];
    }
    delete[] lambda_for_cik;
    delete[] up, lo;
    //-------------------------------------------------------------------------------B & B

    ftime(&endTime);
    cout << "Solve time: " << (endTime.time - startTime.time) * 1000 + (endTime.millitm - startTime.millitm) << endl;
    end();
    return 0;
}

void insert_node(problem current_problem) {
    if (current_problem.ub >= global_lb) {
        int inserted(0);
        double new_lb(current_problem.lb);

        if (problem_list.empty())
            problem_list.push_front(current_problem);
        else {
            list<problem>::iterator it_list = problem_list.begin();
            while (it_list != problem_list.end()) {
                double current_lb((*it_list).lb);

                if (new_lb >= current_lb) {
                    problem_list.insert(it_list, current_problem);
                    it_list = problem_list.end();
                    inserted = 1;
                } else
                    ++it_list;
            }
            if (inserted == 0)
                problem_list.push_back(current_problem);
        }
    }
}

void delete_node() {
    list<problem>::iterator it_list;
    for (it_list = problem_list.begin(); it_list != problem_list.end();) {
        double current_ub((*it_list).ub);
        if (current_ub <= global_lb) {
            it_list = problem_list.erase(it_list);
        } else {
            it_list++;
        }
    }
}

void end() {
    cout << "global_ub: " << global_ub << endl;
    cout << "global_lb: " << global_lb << endl;
    cout << "B&B node: " << nb_solved_node << endl;
}