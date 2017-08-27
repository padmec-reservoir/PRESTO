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
mb.load_file("fine_grid.h5m")

gid_tag = mb.tag_get_handle("GLOBAL_ID_COARSE")  # _COARSE
coarse_perm_tag = mb.tag_get_handle("PRIMAL_PERM")  # PRIMAL_
injection_tag = mb.tag_get_handle("injection_well_coarse")
production_tag = mb.tag_get_handle("production_well_coarse")

injection_sets = mb.get_entities_by_type_and_tag(
    0, types.MBENTITYSET, np.array((injection_tag,)), np.array((None,)))

production_sets = mb.get_entities_by_type_and_tag(
    0, types.MBENTITYSET, np.array((production_tag,)), np.array((None,)))

tag2injection_well = {}
tag2production_well = {}
print "Saving tags..."
for tag in injection_sets:
    tag_id = mb.tag_get_data(injection_tag, np.array([tag]), flat=True)
    elems = mb.get_entities_by_handle(tag, True)
    tag2injection_well[tag_id[0]] = {e for e in elems}

for tag in production_sets:
    tag_id = mb.tag_get_data(production_tag, np.array([tag]), flat=True)

    elems = mb.get_entities_by_handle(tag, True)
    tag2production_well[tag_id[0]] = {e for e in elems}


injection_boundary_cond = {
    1: 1.0
}

production_boundary_cond = {
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0
}

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
percent = 0

for idx, elem in zip(v_ids, volumes):
    count += 1
    if count == (len(volumes) / 100):
        percent += 1
        print percent, "%"
        count = 0

    adj_volumes = mesh_topo_util.get_bridge_adjacencies(
        np.asarray([elem]), 2, 3, 0)
    adj_volumes_set = set(adj_volumes)
    boundary = False

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

        elem_center = mesh_topo_util.get_average_position(np.asarray([elem]))
        K1 = mb.tag_get_data(coarse_perm_tag, [elem], flat=True)

        #adj_perms = mb.tag_get_data(coarse_perm_tag, adj_volumes, flat=True)
        adj_perms = []
        for adjacencies in range(len(adj_volumes)):
            adj_perms.append(mb.tag_get_data(
                     coarse_perm_tag, adj_volumes, flat=True)[
                     adjacencies*9:(adjacencies+1)*9])


        values = []

        for K2, adj in zip(adj_perms, adj_volumes_set):
            adj_center = mesh_topo_util.get_average_position(
                         np.asarray([adj]))
            N = elem_center - adj_center
            N = N / np.sqrt(N[0] ** 2 + N[1] ** 2 + N[2] ** 2)
            K1proj = np.dot(np.dot(N, K1.reshape([3, 3])), N)
            K2proj = np.dot(np.dot(N, K2.reshape([3, 3])), N)
            dl = np.linalg.norm((elem_center - adj_center)/2)
            K_eq = (2 * K1proj * K2proj) / (K1proj * dl + K2proj * dl)
            values.append(- K_eq)

        ids = mb.tag_get_data(gid_tag, adj_volumes)

        values = np.append(values, -(np.sum(values)))

        ids = np.asarray(np.append(ids, idx), dtype='int32')

        A.InsertGlobalValues(idx, values, ids)
        #print idx, ids, values
A.FillComplete()

print "Matrix fill took", time.time() - t0, "seconds... Ran over ", len(volumes), "elems"

mb.tag_set_data(pres_tag, volumes, np.asarray(b[v_ids]))


if USE_DIRECT_SOLVER:
    outfile_template = "Results/output_direct_{0}.vtk"
else:
    outfile_template = "output_iterative_fine_grid{0}.vtk"

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

mb.tag_set_data(pres_tag, volumes, np.asarray(x[v_ids]))

mb.write_file(outfile_template.format(1))
