/*
    TODO:
        - Resolver o sistema encontrado no final.
        - Buscar exemplos e testar desempenho.
            - Procurar ferramentas para medição de desempenho de programas paralelos.
        - Gerar malha com o resultado final.
        - Refatorar código.
            - Modularizar os trechos de código (organizar em funções e classes).
*/


#include "moab/Core.hpp"
#include "moab/MeshTopoUtil.hpp"
#ifdef MOAB_HAVE_MPI
#include "moab/ParallelComm.hpp"
#include "MBParallelConventions.h"
#endif
#include <Epetra_MpiComm.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_Version.h>
#include <AztecOO.h>
#include <mpi.h>
#include <iostream>
#include <string>
#include <set>
#include <numeric>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#define ALL_PROCS -1
#define ALL_DIM -1
#define GHOST_DIM 3
#define BRIDGE_DIM 2

using namespace std;
using namespace moab;

double get_equivalent_perm (double k1[9], double k2[9], double unit_vector[3]) {
    double k1_pre[3] = {k1[0]*unit_vector[0], k1[4]*unit_vector[1], k1[8]*unit_vector[2]};
    double k2_pre[3] = {k2[0]*unit_vector[0], k2[4]*unit_vector[1], k2[8]*unit_vector[2]};
    double k1_eq = k1_pre[0]*unit_vector[0] + k1_pre[1]*unit_vector[1] + k1_pre[2]*unit_vector[2];
    double k2_eq = k2_pre[0]*unit_vector[0] + k2_pre[1]*unit_vector[1] + k2_pre[2]*unit_vector[2];
    return 2*k1_eq*k2_eq/(k1_eq + k2_eq);
}

double get_centroid_dist (double c1[], double c2[]) {
    return sqrt(pow(c1[0] + c2[0], 2) + pow(c1[1] + c2[1], 2) + pow(c1[2] + c2[2], 2));
}

void get_unit_vector (double c1[], double c2[], double dist, double *u) {
    u[0] = (c2[0] - c1[0])/dist;
    u[1] = (c2[1] - c1[1])/dist;
    u[2] = (c2[2] - c1[2])/dist;
}

int main(int argc, char **argv) {
#ifndef MOAB_HAVE_MPI
  cout << "Run with mpiexec or mpirun" << endl;
  return -1;
#endif
    MPI_Init(&argc, &argv);

    // Options for file read.
    string input_file = "part_mesh.h5m";
    string output_file = "solve_mesh.h5m";
    string parallel_read_opts = "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS";
    ErrorCode rval;

    // In case the user provides another file name, use it.
    if (argc > 2) {
        input_file = string(argv[1]);
    }

    // Creating instances of moab::Core and moab::ParallelComm
    Interface* mb = new Core;
    if (mb == NULL) {
        cout << "Unable to create MOAB interface." << endl;
        return -1;
    }

    ParallelComm* pcomm = new ParallelComm(mb, MPI_COMM_WORLD);
    int rank, world_size;
    if (pcomm == NULL) {
        cout << "Unable to create ParallelComm instance." << endl;
        return -1;
    }
    else {
        // Getting processor rank and the total number of processes on communicator.
        rank = pcomm->proc_config().proc_rank();
        world_size = pcomm->proc_config().proc_size();
    }

    // Open file containing the mesh with the options specified above.
    rval = mb->load_file(input_file.c_str(), 0, parallel_read_opts.c_str());
    MB_CHK_SET_ERR(rval, "load_file failed");

    // Creating an instance of moab::MeshTopoUtil for handling operations with
    // elements adjacencies.
    MeshTopoUtil* topo_util = new MeshTopoUtil(mb);
    if (topo_util == NULL) {
        cout << "Failed to create MeshTopoUtil instance." << endl;
        return -1;
    }

    // Exchange one layer of ghost elements, i.e., get the neighbors elements
    // that belong to another partition.
    Range my_elems;
    rval = mb->get_entities_by_dimension(0, 3, my_elems, false); MB_CHK_ERR(rval);
    rval = pcomm->exchange_ghost_cells(GHOST_DIM, BRIDGE_DIM, 1, 0, true);
    MB_CHK_SET_ERR(rval, "exchange_ghost_cells failed");

    // Creating the Epetra structures (Epetra_map and Epetra_comm)
    Epetra_MpiComm epetra_comm (MPI_COMM_WORLD);

    // Computes the number of elements in the mesh.
    int num_local_elems = my_elems.size(), num_global_elems = 0;
    MPI_Allreduce(&num_local_elems, &num_global_elems, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Gets the global ID for each element in the partition.
    int* gids = (int*) calloc(my_elems.size(), sizeof(int));
    if (gids == NULL) {
        cout << "Null pointer" << endl;
        return -1;
    }
    Tag global_id_tag, centroid_tag, perm_tag, dirichlet_tag;
    rval = mb->tag_get_handle("GLOBAL_ID", global_id_tag); MB_CHK_ERR(rval);
    rval = mb->tag_get_handle("CENTROID", centroid_tag); MB_CHK_ERR(rval);
    rval = mb->tag_get_handle("PERMEABILITY", perm_tag); MB_CHK_ERR(rval);
    rval = mb->tag_get_handle("DIRICHLET_BC", dirichlet_tag); MB_CHK_ERR(rval);
    rval = mb->tag_get_data(global_id_tag, my_elems, (void*) gids); MB_CHK_ERR(rval);

    // It is necessary to exchange tags for the ghost elements. Those are not
    // transferred when exchange_ghost_cells is called.
    Range empty_set;
    rval = pcomm->exchange_tags(centroid_tag, empty_set);
    MB_CHK_SET_ERR(rval, "exchange_tags for centroid failed");
    rval = pcomm->exchange_tags(perm_tag, empty_set);
    MB_CHK_SET_ERR(rval, "exchange_tags for permeability failed");
    rval = pcomm->exchange_tags(dirichlet_tag, empty_set);
    MB_CHK_SET_ERR(rval, "exchange_tags for dirichlet bc failed");

    // Initializing the Epetra structure for a sparse matrix.
    Epetra_Map row_map (num_global_elems, num_local_elems, gids, 0, epetra_comm);
    Epetra_CrsMatrix A (Copy, row_map, 6);   // Sparse matrix w/ at most 6 entries per line.
    Epetra_Vector B (row_map);
    Epetra_Vector X (row_map);

    // The code block above does the actual calculation for the method.
    Range adjacencies;
    std::vector<double> row_values;
    std::vector<int> row_indexes;
    double e1_centroid[3], e2_centroid[3], e1_perm[9], e2_perm[9], pressure = 0, diag_coef = 0;
    double equiv_perm = 0.0, centroid_dist = 0.0;
    double unit_vector[3] = {0, 0, 0};
    int row_id = -1, n = 0;
    printf("<%d> Computing the method\n", rank);
    for (Range::iterator it = my_elems.begin(); it != my_elems.end(); it++) {
        rval = mb->tag_get_data(dirichlet_tag, &(*it), 1, &pressure); MB_CHK_ERR(rval);
        if (pressure == 0) {
            rval = topo_util->get_bridge_adjacencies(*it, 2, 3, adjacencies); MB_CHK_ERR(rval);
            rval = mb->tag_get_data(centroid_tag, &(*it), 1, &e1_centroid); MB_CHK_ERR(rval);
            rval = mb->tag_get_data(perm_tag, &(*it), 1, &e1_perm); MB_CHK_ERR(rval);
            for (Range::iterator itt = adjacencies.begin(); itt != adjacencies.end(); itt++) {
                rval = mb->tag_get_data(centroid_tag, &(*itt), 1, &e2_centroid); MB_CHK_ERR(rval);
                rval = mb->tag_get_data(perm_tag, &(*itt), 1, &e2_perm); MB_CHK_ERR(rval);
                rval = mb->tag_get_data(global_id_tag, &(*itt), 1, &row_id); MB_CHK_ERR(rval);

                centroid_dist = get_centroid_dist(e1_centroid, e2_centroid);

                get_unit_vector(e1_centroid, e2_centroid, centroid_dist, &unit_vector[0]);
                equiv_perm = get_equivalent_perm(e1_perm, e2_perm, unit_vector);

                row_values.push_back(-equiv_perm/pow(centroid_dist, 2));
                row_indexes.push_back(row_id);
            }
            diag_coef = (-1)*accumulate(row_values.begin(), row_values.end(), 0.0);
        }
        else {
            diag_coef = 1;
        }
        rval = mb->tag_get_data(global_id_tag, &(*it), 1, &row_id); MB_CHK_ERR(rval);

        row_values.push_back(diag_coef);
        row_indexes.push_back(row_id);

        A.InsertGlobalValues(row_id, row_values.size(), &row_values[0], &row_indexes[0]);
        B[n++] = pressure;

        row_values.clear();
        row_indexes.clear();
        adjacencies.clear();
    }
    printf("<%d> Done.\n", rank);
    A.FillComplete();
    MPI_Barrier(MPI_COMM_WORLD);

    Epetra_LinearProblem linear_problem (&A, &X, &B);
    AztecOO solver (linear_problem);
    solver.Iterate(100000, 1e-10);
    printf("<%d> Solved it.\n", rank);

    printf("<%d> Setting pressure tag\n", rank);
    Tag pressure_tag;
    rval = mb->tag_get_handle("PRESSURE", 1, MB_TYPE_DOUBLE, pressure_tag, MB_TAG_DENSE | MB_TAG_CREAT);
    MB_CHK_SET_ERR(rval, "tag_get_handle to pressure_tag failed");
    rval = mb->tag_set_data(pressure_tag, my_elems, &X[0]);
    MB_CHK_SET_ERR(rval, "tag_set_data to pressure_tag failed");
    printf("<%d> Done\n", rank);

    EntityHandle volumes_meshset;
    rval = mb->create_meshset(0, volumes_meshset);
    MB_CHK_SET_ERR(rval, "create_meshset failed");
    rval = mb->write_file("solved_mesh.h5m", 0, 0, &volumes_meshset, 1);
    MB_CHK_SET_ERR(rval, "write_file failed");

    // Cleaning up alocated objects.
    delete topo_util;
    delete pcomm;
    delete mb;
    MPI_Finalize();

    return 0;
}
