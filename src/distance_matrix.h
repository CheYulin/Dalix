#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>
#include <cstdlib>

using namespace std;

string three_to_one_letter(string tlc);

double **distanceMatrix(string filename1, int *nb_row) {

    string filename(filename1);
    char chain('A');
    double d_threshold(1000);

    vector<int> pdb_aa_number;
    vector<double> coord_x;
    vector<double> coord_y;
    vector<double> coord_z;
    string sse_seq;
    string seq;

    int nb_aa;
    int nb_aa_contact;
    int **aa_contact_map;
    int *degree;

    double **aa_dist_mat;


    ifstream pdb_file;
    pdb_file.open(filename.c_str());
    if (!pdb_file.is_open()) {
        cout << "Unable to open PDB file\n";
        system("pause");
        exit(0);
    }

    char aa_name[16];
    char at_name[16];
    char at_number[16];
    char r_number[16];
    char at_x[16], at_y[16], at_z[16];
    double dij;

    double cx_ca, cy_ca, cz_ca;
    double cx_cb, cy_cb, cz_cb;
    char buffer[1024];
    int ca_number = 1000000;
    int cb_number = 1000000;
    int res_number = 1000000;
    int old_ca_number = 1000000;
    int old_cb_number = 1000000;
    int old_res_number = 1000000;
    bool found_ca = false;
    bool found_cb = false;
    bool read_chain = false;

    while (pdb_file.getline(buffer, 1024)) {

        strncpy(aa_name, buffer + 17, 3);
        aa_name[3] = '\0';

        strncpy(r_number, buffer + 22, 4);
        r_number[4] = '\0';
        res_number = atoi(r_number);

        if (res_number != old_res_number) {
            old_res_number = res_number;
            found_ca = false;
            found_cb = false;
        }

        //Atoms
        if (strncmp(buffer, "ATOM", 4) == 0 || strncmp(buffer, "HETATM", 6) == 0) {
            //Parse line
            strncpy(at_name, buffer + 12, 4);
            at_name[4] = '\0';

            //check if atom is an alpha carbon coming from the correct chain
            if (strncmp(at_name, " CA ", 4) == 0 && buffer[21] == chain) {
                strncpy(at_number, buffer + 22, 4);
                at_number[4] = '\0';
                ca_number = atoi(at_number);

                if (ca_number == old_ca_number) {
                    //alternative positions
                    cout << "alpha_carbon " << ca_number << " is discarded, alternative position\n";
                } else {
                    strncpy(at_x, buffer + 30, 8);
                    at_x[8] = '\0';
                    strncpy(at_y, buffer + 38, 8);
                    at_y[8] = '\0';
                    strncpy(at_z, buffer + 46, 8);
                    at_z[8] = '\0';
                    cx_ca = atof(at_x);
                    cy_ca = atof(at_y);
                    cz_ca = atof(at_z);
                    found_ca = true;
                    old_ca_number = ca_number;
                }
            }

            //check if atom is a beta carbon coming from the correct chain
            if (strncmp(at_name, " CB ", 4) == 0 && buffer[21] == chain) {
                strncpy(at_number, buffer + 22, 4);
                at_number[4] = '\0';
                cb_number = atoi(at_number);

                if (cb_number == old_cb_number) {
                    //alternative positions
                    cout << "beta_carbon " << cb_number << " is discarded, alternative position\n";
                } else {
                    strncpy(at_x, buffer + 30, 8);
                    at_x[8] = '\0';
                    strncpy(at_y, buffer + 38, 8);
                    at_y[8] = '\0';
                    strncpy(at_z, buffer + 46, 8);
                    at_z[8] = '\0';
                    cx_cb = atof(at_x);
                    cy_cb = atof(at_y);
                    cz_cb = atof(at_z);
                    found_cb = true;
                    old_cb_number = cb_number;
                }
            }


            if ((ca_number == cb_number) && found_ca == true && found_cb == true) {
                seq += three_to_one_letter(aa_name);
                sse_seq += "X";
                pdb_aa_number.push_back(ca_number);

                coord_x.push_back(cx_ca);
                coord_y.push_back(cy_ca);
                coord_z.push_back(cz_ca);

                read_chain = true;
                found_ca = false;
                found_cb = false;
            }

            // special case glycin if cb distances are used.
            if (strncmp(aa_name, "GLY", 3) == 0 && found_ca == true) {
                seq += three_to_one_letter(aa_name);
                sse_seq += "X";
                pdb_aa_number.push_back(ca_number);
                coord_x.push_back(cx_ca);
                coord_y.push_back(cy_ca);
                coord_z.push_back(cz_ca);
                read_chain = true;
                found_ca = false;
                found_cb = false;
            }

        } else if (read_chain && strncmp(buffer, "ENDMDL", 6) == 0) {
            break;
        }
    }
    nb_aa = pdb_aa_number.size();
    *nb_row = nb_aa;

    nb_aa_contact = 0;
    //generating inter-amino-acid contact map and the distance matrix
    degree = new int[nb_aa];
    for (int i(0); i < nb_aa; ++i) {
        degree[i] = 0;
    }
    aa_contact_map = new int *[nb_aa];
    aa_dist_mat = new double *[nb_aa];
    for (int i(0); i < nb_aa; ++i) {
        aa_contact_map[i] = new int[nb_aa];
        aa_dist_mat[i] = new double[nb_aa];
        for (int j(0); j < nb_aa; ++j) {
            aa_contact_map[i][j] = 0;
            aa_dist_mat[i][j] = .0;
        }
    }

    for (int i(0); i < nb_aa - 1; ++i) {
        //contact would start at i+2, but distances start at i+1
        for (int j(i + 1); j < nb_aa; ++j) {

            dij = sqrt((coord_x[i] - coord_x[j]) * (coord_x[i] - coord_x[j])
                       + (coord_y[i] - coord_y[j]) * (coord_y[i] - coord_y[j])
                       + (coord_z[i] - coord_z[j]) * (coord_z[i] - coord_z[j]));

            aa_dist_mat[i][j] = dij;
            aa_dist_mat[j][i] = dij;
            if (dij < d_threshold) {
                aa_contact_map[i][j] = 1;
                aa_contact_map[j][i] = 1;
                degree[i] += 1;
                degree[j] += 1;
                nb_aa_contact++;
            }
        }
    }

    return aa_dist_mat;
}

string three_to_one_letter(string tlc) {
    if (tlc == "ALA" || tlc == "CYS" || tlc == "GLY" || tlc == "HIS" || tlc == "ILE" || tlc == "LEU" || tlc == "MET" ||
        tlc == "PRO" || tlc == "SER" || tlc == "THR" || tlc == "VAL") {
        return tlc.substr(0, 1);
    } else if (tlc == "ARG" || tlc == "TYR") return tlc.substr(1, 1);
    else if (tlc == "ASN") return tlc.substr(2, 1);
    else if (tlc == "ASP") return "D";
    else if (tlc == "GLU") return "E";
    else if (tlc == "GLN") return "Q";
    else if (tlc == "LYS") return "K";
    else if (tlc == "PHE") return "F";
    else if (tlc == "TRP") return "W";
    else return "X";
}