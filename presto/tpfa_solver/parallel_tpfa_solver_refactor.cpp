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

class TPFASolver {
    Interface *mbcore;
    ParallelComm *pcomm;
    MeshTopoUtil *topo_util;
    char *perm_tag_name;
    char *centroid_tag_name;
    char *dirichlet_tag_name;
    char *neumann_tag_name;
public:
    TPFASolver ();
    TPFASolver (Interface*, char*, char*, char*, char*);
    virtual ~TPFASolver ();
    void run_method ();
private:
    double calculate_equivalent_perm (double k1[9], double k2[9], double u[3]);
    double calculate_centroid_dist (double c1[3], double c2[3]);
    double* calculate_unit_vector (double c1[3], double c2[3]);
};

TPFASolver::TPFASolver () {
    this->mbcore = new Core ();
    this->pcomm = new ParallelComm (this->mbcore, MPI_COMM_SELF);
    this->topo_util = new MeshTopoUtil (mbcore);
    this->perm_tag_name = "PERMEABILITY";
    this->centroid_tag_name = "CENTROID";
    this->dirichlet_tag_name = "DIRICHLET_BC";
    this->neumann_tag_name = "NEUMANN_BC";
}

TPFASolver::TPFASolver (Interface *mbcore, char* perm_tag_name, char *centroid_tag_name, 
                        char* dirichlet_tag_name, char* neumann_tag_name) {
    this->mbcore = mbcore;
    this->pcomm = new ParallelComm (this->mbcore, MPI_COMM_WORLD);
    this->topo_util = new MeshTopoUtil (this->mbcore);
    this->perm_tag_name = perm_tag_name;
    this->centroid_tag_name = centroid_tag_name;
    this->dirichlet_tag_name = dirichlet_tag_name;
    this->neumann_tag_name = neumann_tag_name;
}

TPFASolver::~TPFASolver () {
    delete this->mbcore;
    delete this->pcomm;
    delete this->topo_util;
    free(this->perm_tag_name);
    free(this->dirichlet_tag_name);
    free(this->neumann_tag_name);
}

void TPFASolver::run_method () {
    return;
}

double TPFASolver::calculate_equivalent_perm (double k1[9], double k2[9], double u[3]) {
    double k1_pre[3] = {k1[0]*u[0], k1[4]*u[1], k1[8]*u[2]};
    double k2_pre[3] = {k2[0]*u[0], k2[4]*u[1], k2[8]*u[2]};
    double k1_post = k1_pre[0]*u[0] + k1_pre[1]*u[1] + k1_pre[2]*u[2];
    double k2_post = k2_pre[0]*u[0] + k2_pre[1]*u[1] + k2_pre[2]*u[2];
    return 2*k1_post*k2_post/(k1_post + k2_post);
}

double TPFASolver::calculate_centroid_dist (double c1[3], double c2[3]) {
    return std::sqrt(std::pow(c1[0] + c2[0], 2) + std::pow(c1[1] + c2[1], 2) + std::pow(c1[2] + c2[2], 2));
}

double* TPFASolver::calculate_unit_vector (double c1[3], double c2[3]) {
    double dist = this->calculate_centroid_dist(c1, c2);
    double u[3] = {(c2[0] - c1[0])/dist, (c2[1] - c1[1])/dist, (c2[2] - c1[2])/dist};
    return u;
}

int main(int argc, char const *argv[])
{

    return 0;
}
