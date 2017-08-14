import time

from pymoab import core
from pymoab import types
from pymoab import topo_util
import numpy as np
from PyTrilinos import Epetra, AztecOO, Amesos

USE_DIRECT_SOLVER = False

comm = Epetra.PyComm()
mb = core.Core()
root_set = mb.get_root_set()
mesh_topo_util = topo_util.MeshTopoUtil(mb)

print "Loading..."
mb.load_file("coarse_mesh.h5m")

gid_tag = mb.tag_get_handle("PRIMAL_GLOBAL_ID")
coarse_perm_tag = mb.tag_get_handle("PRIMAL_PERM")
finals_in_primal = mb.tag_get_handle("FINE_TO_PRIMAL")

volumes = mb.get_entities_by_dimension(0, 3)
perm_values = mb.tag_get_data(coarse_perm_tag, volumes)
v_ids = mb.tag_get_data(gid_tag, volumes).flatten()
v_ids = np.subtract(v_ids, np.min(v_ids))

std_map = Epetra.Map(len(volumes), 0, comm)
A = Epetra.CrsMatrix(Epetra.Copy, std_map, 0)

# Initial volume pressure data
pres_tag = mb.tag_get_handle(
    "Pressure", 1, types.MB_TYPE_DOUBLE,
    types.MB_TAG_SPARSE, True)
b = Epetra.Vector(std_map)
x = Epetra.Vector(std_map)
mb.tag_set_data(pres_tag, volumes, np.asarray(b))

print "Filling matrix..."
t0 = time.time()
count = 0
verts_temp = []
for idx, elem in zip(v_ids, volumes):
    count += 1
    adj_volumes = mesh_topo_util.get_bridge_adjacencies(
        np.asarray([elem]), 2, 3)
    adj_volumes_set = set(adj_volumes)
    print adj_volumes_set

    boundary = False
    """
    for tag, well_elems in tag2injection_well.iteritems():
        if elem in well_elems:
            b[idx] = injection_boundary_cond[tag]
            boundary = True

    for tag, well_elems in tag2production_well.iteritems():
        if elem in well_elems:
            b[idx] = production_boundary_cond[tag]
            boundary = True


    if boundary:
        A.InsertGlobalValues(idx, [1], [idx])

    if not boundary:

        el1_center = mesh_topo_util.get_average_position(np.asarray([elem]))
        K1 = mb.tag_get_data(coarse_perm_tag, [elem], flat=True)[0]

        #adj_perms = mb.tag_get_data(coarse_perm_tag, adj_volumes, flat=True)
        adj_perms = []
        for adjacencies in range(len(adj_volumes)):
            adj_perms.append(mb.tag_get_data(coarse_perm_tag,
            adj_volumes, flat=True)[adjacencies*9]
            )

        values = []

        for K2, adj in zip(adj_perms, adj_volumes_set):
            el2_center = mesh_topo_util.get_average_position(np.asarray([adj]))
            dx = np.linalg.norm((el1_center - el2_center)/2)

            K_equiv = (2*K1*K2) / (K1*dx + K2*dx)
            values.append(-K_equiv)

        ids = mb.tag_get_data(gid_tag, adj_volumes)

        values = np.append(values, -(np.sum(values)))

        ids = np.asarray(np.append(ids, idx), dtype='int32')

        A.InsertGlobalValues(idx, values, ids)
        #print idx, ids, values
print b
A.FillComplete()

print "Matrix fill took", time.time() - t0, "seconds... Ran over ", len(volumes), "elems"

mb.tag_set_data(pres_tag, volumes, np.asarray(b[v_ids]))


if USE_DIRECT_SOLVER:
    outfile_template = "Results/output_direct_{0}.vtk"
else:
    outfile_template = "Results/output_iterative_hmean{0}.vtk"

mb.write_file(outfile_template.format(0))

linearProblem = Epetra.LinearProblem(A, x, b)

if USE_DIRECT_SOLVER:
    solver = Amesos.Lapack(linearProblem)

    print "1) Performing symbolic factorizations..."
    solver.SymbolicFactorization()

    print "2) Performing numeric factorizations..."
    solver.NumericFactorization()
    print "3) Solving the linear system..."
    ierr = solver.Solve()
else:

    solver = AztecOO.AztecOO(linearProblem)

    ierr = solver.Iterate(1000, 1e-9)

print "   solver.Solve() return code = ", ierr

if ierr <= 1e-9:

    print
    print "|--------------------|"
    print "|convergence achieved|"
    print "|--------------------|"
    print

np.savetxt('coarse_pressure', np.asarray(x), delimiter = "\t")



mb.tag_set_data(pres_tag, volumes, np.asarray(x[v_ids]))

mb.write_file(outfile_template.format(1))

"""
