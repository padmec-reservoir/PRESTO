#!/usr/bin/env python3
import sys
import time
import numpy as np
from pymoab import core, types, rng
from pymoab.tag import Tag

# Global variables with default values.
nx = 10     # Number of elements along the x axis.
ny = 3      # Number of elements along the y axis.
nz = 1      # Number of elements along the z axis.
dx = 2.0    # Element size along the x axis.
dy = 5/3    # Element size along the y axis.
dz = 1      # Element size along the z axis.
num_elements = 30


# create_mesh_connectivity: Defines the vertices that make an element.
# Parameters:
#   - vertex_handles: NumPy array of entiites handles for the vertices
#   - vertex_coords: NumPy array containing vertices coordinates.
def create_mesh_connectivity(vertex_handles, vertex_coords):
    global nx, ny, nz, num_elements

    k = 0
    mesh_connectivity = np.zeros((num_elements, 8), dtype=np.uint64)

    # Defining the vertices which can start an element, i.e., starting from this
    # vertex one can get five more valid vertices to build an element.
    indexes = [v-1 for v in vertex_handles
                    if (vertex_coords[3*(v-1)] != nx*dx) and \
                        (vertex_coords[3*(v-1)+1] != ny*dy) and \
                        (vertex_coords[3*(v-1)+2] != nz*dz)]
    for i in indexes:
        mesh_connectivity[k] = [vertex_handles[i], vertex_handles[i+1],\
                                vertex_handles[i+nx+2], vertex_handles[i+nx+1],\
                                vertex_handles[i+(nx+1)*(ny+1)], vertex_handles[i+(nx+1)*(ny+1)+1],\
                                vertex_handles[i+(nx+1)*(ny+2)+1], vertex_handles[i+(nx+1)*(ny+2)]]
        k += 1

    return mesh_connectivity

def main():
    global nx, ny, nz, dx, dy, dz, num_elements

    # Getting user input for number of elements and size in each axis.
    if len(sys.argv) == 7:
        nx = int(sys.argv[1])
        ny = int(sys.argv[3])
        nz = int(sys.argv[5])
        dx = float(sys.argv[2])
        dy = float(sys.argv[4])
        dz = float(sys.argv[6])
        dim = 2
        num_elements = nx*ny*nz
        num_vertex = (nx+1)*(ny+1)*(nz+1)
    else:
        print("Not enough arguments")
        return

    # New instance of MOAB Core
    mbcore = core.Core()

    # Initializing array with vertices coordinates. The array looks like
    # [x y z x y z ...].
    print("Creating vertices coordinates")
    ts = time.time()
    vertex_coords = np.zeros(num_vertex*3)
    for i in range(num_vertex):
        vertex_coords[3*i] = (i % (nx+1))*dx
        vertex_coords[3*i+1] = ((i // (nx+1)) % (ny+1))*dy
        vertex_coords[3*i+2] = ((i // ((nx+1)*(ny+1))) % (nz+1))*dz
    print("Done\nTime elapsed: {0}\n".format(time.time() - ts))

    # Create entity handles for the vertices.
    vertex_handles = mbcore.create_vertices(vertex_coords)

    # Getting the connectivity of each element, i.e., the vertices that make up
    # an element.
    print("Creating connectivity")
    ts = time.time()
    mesh_connectivity = create_mesh_connectivity(vertex_handles, vertex_coords)
    print("Done\nTime elapsed: {0}\n".format(time.time() - ts))

    # Creates the element corresponding to each connectivity.
    print("Creating element handles")
    ts = time.time()
    elem_handles = rng.Range([mbcore.create_element(types.MBHEX, x) for x in mesh_connectivity])
    print("Done\nTime elapsed: {0}\n".format(time.time() - ts))

    # Setting up tags for permeability and centroid coordinates for each element.
    print("Creating and setting tags")
    ts = time.time()
    centroid_tag = mbcore.tag_get_handle('Centroid', 3, types.MB_TYPE_DOUBLE, \
                                          types.MB_TAG_DENSE, True)
    permeability_tag = mbcore.tag_get_handle('Permeability', 3, types.MB_TYPE_DOUBLE, \
                                              types.MB_TAG_DENSE, True)
    dirichlet_tag = mbcore.tag_get_handle('DirichletBC', 1, types.MB_TYPE_DOUBLE, \
                                            types.MB_TAG_DENSE, True)
    neumann_tag = mbcore.tag_get_handle('NeumannBC', 1, types.MB_TYPE_DOUBLE, \
                                        types.MB_TAG_DENSE, True)

    centroid_coord = np.array([[vertex_coords[3*int(v[0]-1)] + (dx/2), \
                                vertex_coords[3*int(v[0]-1)+1] + (dy/2), \
                                vertex_coords[3*int(v[0]-1)+2] + (dz/2)] \
                                for v in mesh_connectivity])
    permeability = np.array([[1.0,1.0,1.0] for i in range(num_elements)])

    mbcore.tag_set_data(centroid_tag, elem_handles, centroid_coord)
    mbcore.tag_set_data(permeability_tag, elem_handles, permeability)
    print("Done\nTime elapsed: {0}\n".format(time.time() - ts))

    print("Writing .h5m file")
    mbcore.write_file("tpfa_mesh.h5m")
    ts = time.time()
    print("Done\nTime elapsed: {0}\n".format(time.time() - ts))


if __name__ == '__main__':
    main()
