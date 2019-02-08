/*
    TODO:
    - Usar o parmetis para fazer o particionamento em paralelo.
    - Fazer o particionamento chamando o metis diretamente.
    - Adicionar ao write_file o meshset para retirar erro da visualização.
*/

#include "moab/Core.hpp"
#include "moab/MetisPartitioner.hpp"
#ifdef MOAB_HAVE_MPI
#include "moab/ParallelComm.hpp"
#include "MBParallelConventions.h"
#endif
#include <iostream>
#include <string>

using namespace std;
using namespace moab;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    ErrorCode rval;
    string input_file = "tpfa_mesh.h5m";
    string output_file = "part_mesh.h5m";

    Interface* mb = new Core;
    if (mb == NULL) {
        cout << "Unable to create MOAB interface." << endl;
        return -1;
    }

    MetisPartitioner* metis_tool = new MetisPartitioner(mb);
    if (metis_tool == NULL) {
        cout << "Unable to create metis partitioner instance." << endl;
        return -1;
    }

    rval = mb->load_file(input_file.c_str());
    MB_CHK_SET_ERR(rval, "load_file failed");

    rval = metis_tool->partition_mesh((idx_t) 4, "ML_KWAY", 3, true, true);
    MB_CHK_SET_ERR(rval, "partition_mesh failed");

    rval = mb->write_file(output_file.c_str());
    MB_CHK_SET_ERR(rval, "write_file failed");

    delete metis_tool;
    delete mb;
    MPI_Finalize();

    return 0;
}
