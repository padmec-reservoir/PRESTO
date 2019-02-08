/* C++ STL includes */
#include <iostream>	/* std::cout, std::cin */
#include <numeric>	/* std::accumulate */
#include <cstdlib>	/* calloc, free */
#include <cstdio>	/* printf */
#include <cmath>	/* sqrt, pow */
#include <ctime>
#include <string>

/* MOAB includes */
#include "moab/Core.hpp"
#include "moab/MeshTopoUtil.hpp"
#ifdef MOAB_HAVE_MPI
#include "moab/ParallelComm.hpp"
#include "MBParallelConventions.h"
#endif

/* Trilinos includes */
#include <Epetra_MpiComm.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_Version.h>
#include <AztecOO.h>

/* MPI header */
#include <mpi.h>

#define ALL_PROCS -1
#define ALL_DIM -1
#define GHOST_DIM 3
#define BRIDGE_DIM 2

using namespace std;
using namespace moab;

// Enumeration created to make the access to tags more readable.
enum TagsID {global_id, permeability, centroid, dirichlet, neumann};

class TPFASolver {
public:
    Interface *mb;
    ParallelComm *pcomm;
    MeshTopoUtil *topo_util;
    MPI_Comm comm;
    char *perm_tag_name;
    char *centroid_tag_name;
    char *dirichlet_tag_name;
    char *neumann_tag_name;

    TPFASolver () {
        /* Default constructor */
        this->mb = new Core ();
        this->comm = MPI_COMM_WORLD;
        this->pcomm = new ParallelComm (this->mb, this->comm);
        this->topo_util = new MeshTopoUtil (mb);
        this->perm_tag_name = "PERMEABILITY";
        this->centroid_tag_name = "CENTROID";
        this->dirichlet_tag_name = "DIRICHLET_BC";
        this->neumann_tag_name = "NEUMANN_BC";
    }

    TPFASolver (Interface *mb, MPI_Comm comm, char* perm_tag_name, char *centroid_tag_name,
	            char* dirichlet_tag_name, char* neumann_tag_name) {
	    /* Main constructor */
        this->mb = mb;
        this->comm = comm;
        this->pcomm = new ParallelComm (this->mb, this->comm);
        this->topo_util = new MeshTopoUtil (this->mb);
        this->perm_tag_name = perm_tag_name;
        this->centroid_tag_name = centroid_tag_name;
        this->dirichlet_tag_name = dirichlet_tag_name;
        this->neumann_tag_name = neumann_tag_name;
    }

    virtual ~TPFASolver () {
        /* Destructor */
        free(this->perm_tag_name);
        free(this->dirichlet_tag_name);
        free(this->neumann_tag_name);
    }

    ErrorCode run () {
    	/*
    		Run solver for TPFA problem specificed at given moab::Core
    		instance.

    		Parameters
    		----------
    		None

    		Returns
    		-------
    		MOAB error code
    	*/

        ErrorCode rval;
        clock_t ts;
        int rank = this->pcomm->proc_config().proc_rank();

        // Get all volumes in the mesh and exchange those shared with
        // others processors.
        Range volumes;
        rval = this->mb->get_entities_by_dimension(0, 3, volumes, false);
        MB_CHK_SET_ERR(rval, "get_entitites_by_dimension failed");
        rval = this->pcomm->exchange_ghost_cells(GHOST_DIM, BRIDGE_DIM, 1, 0, true);
        MB_CHK_SET_ERR(rval, "exchange_ghost_cells failed");

        // Calculate the total numbers of elements in the mesh.
        int num_local_elems = volumes.size(), num_global_elems = 0;
        MPI_Allreduce(&num_local_elems, &num_global_elems, 1, MPI_INT, MPI_SUM, this->comm);
        printf("<%d> # global elems: %d\n", rank, num_global_elems);

        // Get tags on elements and exchange those from shared elements.
        Tag tag_handles[5];
        printf("<%d> Setting tags...\n", rank);
        ts = clock();
        this->setup_tags(tag_handles);
        ts = clock() - ts;
        printf("<%d> Done. Time elapsed: %f\n", rank, ((double) ts)/CLOCKS_PER_SEC);
        int* gids = (int*) calloc(volumes.size(), sizeof(int));
        if (gids == NULL) {
            printf("<%d> Error: Null pointer\n", rank);
            exit(EXIT_FAILURE);
        }
        rval = this->mb->tag_get_data(tag_handles[global_id], volumes, (void*) gids);
        MB_CHK_SET_ERR(rval, "tag_get_data for gids failed");

        // Setting up Epetra structures
        Epetra_MpiComm epetra_comm (this->comm);
        Epetra_Map row_map (num_global_elems, num_local_elems, gids, 0, epetra_comm);
        Epetra_CrsMatrix A (Copy, row_map, 7);
        Epetra_Vector b (row_map);
        Epetra_Vector X (row_map);

        printf("<%d> Matrix assembly...\n", rank);
        ts = clock();
        this->assembly_matrix(A, b, volumes, tag_handles);
        ts = clock() - ts;
        printf("<%d> Done. Time elapsed: %f\n", rank, ((double) ts)/CLOCKS_PER_SEC);

        Epetra_LinearProblem linear_problem (&A, &X, &b);
        AztecOO solver (linear_problem);
        solver.Iterate(10000, 1e-14);

        printf("<%d> Setting pressure...\n", rank);
        ts = clock();
        this->set_pressure_tags(X, volumes);
        ts = clock() - ts;
        printf("<%d> Done. Time elapsed: %f\n", rank, ((double) ts)/CLOCKS_PER_SEC);

        free(gids);
        return MB_SUCCESS;
    }

private:
    double calculate_centroid_dist (double c1[3], double c2[3]) {
    	/*
    		Calculate the distance between two points.

    		Parameters
    		----------
    		c1: double*
    			An array with the coordinates of the first point.
    		c2: double*
    			An array with the coordinates of the second point.

    		Returns
    		-------
    		Double value of the distance between c1 and c2.
    	*/

        return sqrt(pow(c1[0] - c2[0], 2) + pow(c1[1] - c2[1], 2) + pow(c1[2] - c2[2], 2));
    }

    double calculate_equivalent_perm (double k1[9], double k2[9], double u[3]) {
	    /*
	    	Calculate the equivalent permeability between k1 and k2. To obtain the
	    	equivalent permeability for each element, the permeability tensor is
	    	multiplied (using the inner product) by the unit vector u twice, i.e,
	    	K1_eq = <<K1, u>, u>, where <,> denotes the inner product.

	    	Parameters
	    	----------
	    	k1: double*
	    		An array representing the permeability tensor of the first element.
	    	k2: double*
		    	An array representing the permeability tensor of the second element.
		    u: double*
		    	An array representing the coordinates of the unit vector.

		    Returns
		    -------
		    Double value of the equivalent permeability of the two elements.
	    */

        double k1_pre[3] = {k1[0]*u[0] + k1[3]*u[0] + k1[6]*u[0],
		                    k1[1]*u[1] + k1[4]*u[1] + k1[7]*u[1],
		                    k1[2]*u[2] + k1[5]*u[2] + k1[8]*u[2]};
        double k2_pre[3] = {k2[0]*u[0] + k2[3]*u[0] + k2[6]*u[0],
		                    k2[1]*u[1] + k2[4]*u[1] + k2[7]*u[1],
		                    k2[2]*u[2] + k2[5]*u[2] + k2[8]*u[2]};
		double k1_eq = k1_pre[0]*u[0] + k1_pre[1]*u[1] + k1_pre[2]*u[2];
		double k2_eq = k2_pre[0]*u[0] + k2_pre[1]*u[1] + k2_pre[2]*u[2];
		return 2*k1_eq*k2_eq/(k1_eq + k2_eq);
    }

    double* calculate_unit_vector (double c1[3], double c2[3]) {
    	/*
    		Calculate the unit vector given two points.

    		Parameters
    		----------
    		c1: double*
    			An array with the coordinates of the first point.
    		c2: double*
    			An array with the coordinates of the second point.

    		Returns
    		-------
    		Array with the vector coordinates.
    	*/

        double d = this->calculate_centroid_dist(c1, c2);
        static double u[3] = {(c2[0] - c1[0])/d, (c2[1] - c1[1])/d, (c2[2] - c1[2])/d};
        return u;
    }

    ErrorCode setup_tags (Tag tags[5]) {
    	/*
    		Construct handles for tags and exchange tags for shared entities.

    		Parameters
    		----------
    		tags: Tag*
				Array that will store the handles.

    		Returns
    		-------
    		MOAB error code.
    	*/

        ErrorCode rval;

        rval = this->mb->tag_get_handle("GLOBAL_ID", tags[global_id]); MB_CHK_ERR(rval);
        rval = this->mb->tag_get_handle(this->centroid_tag_name, tags[centroid]); MB_CHK_ERR(rval);
        rval = this->mb->tag_get_handle(this->perm_tag_name, tags[permeability]); MB_CHK_ERR(rval);
        rval = this->mb->tag_get_handle(this->dirichlet_tag_name, tags[dirichlet]); MB_CHK_ERR(rval);
        rval = this->mb->tag_get_handle(this->neumann_tag_name, tags[neumann]); MB_CHK_ERR(rval);

        std::vector<Tag> tags_vector;
        Range empty_set;
        for (int i = 0; i < 5; i++) {
            tags_vector.push_back(tags[i]);
        }
        rval = this->pcomm->exchange_tags(tags_vector, tags_vector, empty_set);
        MB_CHK_SET_ERR(rval, "exchange_tags failed");

        return MB_SUCCESS;
    }

    ErrorCode assembly_matrix (Epetra_CrsMatrix& A, Epetra_Vector& b, Range volumes, Tag* tag_handles) {
    	/*
    		Assembly the transmissibility matrix, a.k.a, the coeficient matrix A of the linear
    		system to be solved.

    		Parameters
    		----------
    		A: Epetra_CrsMatrix
    			Epetra matrix used to store the coeficients.
    		b: Epetra_Vector*
    			Epetra vector used to store the boundary values.
    		volumes: moab::Range
    			MOAB Range containing the entity handles for all volumes.
    		tag_handles: Tag*
    			Array of tag handles

    		Returns
    		-------
    		MOAB error code.
    	*/

        ErrorCode rval;
        Range adjacencies;
        std::vector<double> row_values;
        std::vector<int> row_indexes;
        double c1[3], c2[3], k1[9], k2[9], *u;
        double equiv_perm = 0.0, centroid_dist = 0.0, pressure_bc = 0, flux_bc = 0, diag_coef = 0;
        int row_id = -1, n = 0;

        for (Range::iterator it = volumes.begin(); it != volumes.end(); it++) {
            rval = this->mb->tag_get_data(tag_handles[dirichlet], &(*it), 1, &pressure_bc); MB_CHK_ERR(rval);
            rval = this->mb->tag_get_data(tag_handles[neumann], &(*it), 1, &flux_bc); MB_CHK_ERR(rval);
            if (pressure_bc != 0) {
                diag_coef = 1;
            }
            else {
                rval = this->topo_util->get_bridge_adjacencies(*it, BRIDGE_DIM, 3, adjacencies); MB_CHK_ERR(rval);
                rval = this->mb->tag_get_data(tag_handles[centroid], &(*it), 1, &c1); MB_CHK_ERR(rval);
                rval = this->mb->tag_get_data(tag_handles[permeability], &(*it), 1, &k1); MB_CHK_ERR(rval);
                for (Range::iterator itt = adjacencies.begin(); itt != adjacencies.end(); itt++) {
                    rval = this->mb->tag_get_data(tag_handles[centroid], &(*itt), 1, &c2); MB_CHK_ERR(rval);
                    rval = this->mb->tag_get_data(tag_handles[permeability], &(*itt), 1, &k2); MB_CHK_ERR(rval);
                    rval = this->mb->tag_get_data(tag_handles[global_id], &(*itt), 1, &row_id); MB_CHK_ERR(rval);
                    centroid_dist = this->calculate_centroid_dist(c1, c2);
                    u = this->calculate_unit_vector(c1, c2);
                    equiv_perm = this->calculate_equivalent_perm(k1, k2, u);
                    row_values.push_back(-equiv_perm/centroid_dist);
                    row_indexes.push_back(row_id);
                }
                diag_coef = -accumulate(row_values.begin(), row_values.end(), 0.0);
            }
            rval = this->mb->tag_get_data(tag_handles[global_id], &(*it), 1, &row_id); MB_CHK_ERR(rval);
            row_values.push_back(diag_coef);
            row_indexes.push_back(row_id);
            A.InsertGlobalValues(row_id, row_values.size(), &row_values[0], &row_indexes[0]);
            if (flux_bc != -1)
                b[n++] = pressure_bc + flux_bc;
            else
                b[n++] = pressure_bc;
            row_values.clear();
            row_indexes.clear();
            adjacencies.clear();
        }
        A.FillComplete();
        return MB_SUCCESS;
    }

    ErrorCode set_pressure_tags (Epetra_Vector& X, Range& volumes) {
    	/*
    		Create and set values for the pressure on each volume.

    		Parameters
    		----------
    		X: Epetra_Vector
    			Epetra vector containing the pressure values for each volume.
    		volumes: moab::Range
    			MOAB Range containing the entity handles for all volumes.

			Returns
			-------
			MOAB error code.
    	*/

        Tag pressure_tag;
        ErrorCode rval;
        rval = this->mb->tag_get_handle("PRESSURE", 1, MB_TYPE_DOUBLE, pressure_tag, MB_TAG_DENSE | MB_TAG_CREAT);
        MB_CHK_SET_ERR(rval, "tag_get_handle for pressure tag failed");
        rval = this->mb->tag_set_data(pressure_tag, volumes, &X[0]);
        MB_CHK_SET_ERR(rval, "tag_set_data for pressure tag failed");
        return MB_SUCCESS;
    }
};


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    TPFASolver* solver = new TPFASolver();
    ErrorCode rval;
    string input_file = "part_mesh.h5m";
    string output_file = "solve_mesh.h5m";
    string parallel_read_opts = "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS";
    string parallel_write_opts = "PARALLEL=WRITE_PART";

    rval = solver->mb->load_file(input_file.c_str(), 0, parallel_read_opts.c_str());
    MB_CHK_SET_ERR(rval, "load_file failed");

    solver->run();

    EntityHandle volumes_meshset;
    rval = solver->mb->create_meshset(0, volumes_meshset);
    MB_CHK_SET_ERR(rval, "create_meshset failed");
    Range my_elems;
    rval = solver->mb->get_entities_by_dimension(0, 3, my_elems, false);
    rval = solver->mb->add_entities(volumes_meshset, my_elems);
    MB_CHK_SET_ERR(rval, "add_entitites failed");

    printf("Writing file\n");
    rval = solver->mb->write_file(output_file.c_str(), 0, parallel_write_opts.c_str(), &volumes_meshset, 1);
    printf("Done\n");
    MB_CHK_SET_ERR(rval, "write_file failed");

    MPI_Finalize();

    return 0;
}
