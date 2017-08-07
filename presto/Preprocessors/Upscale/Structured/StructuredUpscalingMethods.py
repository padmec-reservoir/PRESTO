import numpy as np
from pymoab import types
from pymoab import topo_util


class StructuredUpscalingMethods:
    """Defines a structured upscaling mesh representation
    Parameters
    ----------
    coarse_ratio: List or array of integers
        List or array containing three values indicating the coarsening ratio
        of the mesh in x, y and z directions.
        mesh_size: List or array of integers
            List or array containing three values indicating the mesh size
            (number of fine elements) of the mesh in x, y and z.
        block_size List o array of floats
            List or array containing three values indicating the constant
            increments of vertex coordinates in x, y and z.
        """
    def __init__(self, coarse_ratio, mesh_size, block_size, moab):

        self.coarse_ratio = coarse_ratio
        self.mesh_size = mesh_size
        self.block_size = block_size

        self.verts = None  # Array containing MOAB vertex entities
        self.elems = []  # List containing MOAB volume entities

        self.coarse_verts = None  # Array containing MOAB vertex entities for
        #                           the coarse mesh
        self.coarse_elems = []  # List containig MOAB volume entities for the
        #                         coarse mesh

        self.primals = {}  # Mapping from tuples (idx, dy, idz) to Coarse
        #                    volumes
        self.primal_ids = []

        self.primals_adj = []

        self.perm = []

        # MOAB boilerplate
        self.mb = moab
        self.root_set = self.mb.get_root_set()
        self.mesh_topo_util = topo_util.MeshTopoUtil(self.mb)

    def create_tags(self):
        # TODO: - Should go on Common (?)

        self.gid_tag = self.mb.tag_get_handle(
            "GLOBAL_ID", 1, types.MB_TYPE_INTEGER,
            types.MB_TAG_DENSE, True)

        self.coarse_gid_tag = self.mb.tag_get_handle(
            "PRIMAL_GLOBAL_ID", 1, types.MB_TYPE_INTEGER,
            types.MB_TAG_DENSE, True)

        # this will gide through the meshsets corresponding to coarse scale
        # volumes
        self.primal_id_tag = self.mb.tag_get_handle(
            "PRIMAL_ID", 1, types.MB_TYPE_INTEGER,
            types.MB_TAG_SPARSE, True)

        self.phi_tag = self.mb.tag_get_handle(
            "PHI", 1, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

        self.boundary_tag = self.mb.tag_get_handle(
            "LOCAL BOUNDARY CONDITIONS", 1, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

        # tag handle for upscaling operation
        self.primal_phi_tag = self.mb.tag_get_handle(
            "PRIMAL_PHI", 1, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

        self.perm_tag = self.mb.tag_get_handle(
            "PERM", 9, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

        # tag handle for upscaling operation
        self.primal_perm_tag = self.mb.tag_get_handle(
            "PRIMAL_PERM", 9, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

        # either shoud go or put other directions..., I...
        self.abs_perm_x_tag = self.mb.tag_get_handle(
            "ABS_PERM_X", 1, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

        self.fine_to_primal_tag = self.mb.tag_get_handle(
            "FINE_TO_PRIMAL", 1, types.MB_TYPE_HANDLE,
            types.MB_TAG_SPARSE, True)

        self.primal_adj_tag = self.mb.tag_get_handle(
            "PRIMAL_ADJ", 1, types.MB_TYPE_HANDLE,
            types.MB_TAG_SPARSE, True)

        self.collocation_point_tag = self.mb.tag_get_handle(
            "COLLOCATION_POINT", 1, types.MB_TYPE_HANDLE,
            types.MB_TAG_SPARSE, True)

    def get_block_size_coarse(self):
        block_size_coarse = []
        total_size = (np.asarray(self.mesh_size, dtype='int32')) * np.asarray(
            self.block_size, dtype='float64')

        for dim in range(0, 3):
            block_size_coarse.append([self.coarse_ratio[dim]*np.asarray(
                self.block_size[dim], dtype='float64') * coarse_dim
                for coarse_dim in np.arange(self._coarse_dims()[dim],
                                            dtype='int32')])
            block_size_coarse[dim].append(total_size[dim])
        return block_size_coarse

    def create_coarse_vertices(self):
        # TODO: - Should go on Common

        block_size_coarse = self.get_block_size_coarse()

        coarse_coords = np.array([
            (i, j, k)
            for k in (np.array(block_size_coarse[2], dtype='float64'))
            for j in (np.array(block_size_coarse[1], dtype='float64'))
            for i in (np.array(block_size_coarse[0], dtype='float64'))
            ])

        return self.mb.create_vertices(coarse_coords.flatten())

    def _coarse_dims(self,):
        # TODO: - Should go on Common

        mesh_size_coarse = np.asarray(
            self.mesh_size, dtype='int32') // np.asarray(
                self.coarse_ratio, dtype='int32')
        return mesh_size_coarse

    def calculate_primal_ids(self):
        # TODO: - Should go on Common
        for dim in range(0, 3):
            self.primal_ids.append(
                [i // (self.coarse_ratio[dim]) for i in xrange(
                        self.mesh_size[dim])])

        new_primal = []
        for dim in range(0, 3):
            new_primal.append(
                self.primal_ids[dim][(
                    self.mesh_size[dim] // self.coarse_ratio[dim]) *
                                     self.coarse_ratio[dim]:])

            if len(new_primal[dim]) < (self.mesh_size[dim] // 2):
                new_primal[dim] = np.repeat(
                    max(self.primal_ids[dim]) - 1,
                    len(new_primal[dim])).tolist()
                self.primal_ids[dim] = (self.primal_ids[dim][:self.mesh_size[
                    dim] // self.coarse_ratio[dim] * self.coarse_ratio[dim]] +
                                        new_primal[dim])

    def create_fine_vertices(self):
        # TODO: - Should go on Common

        coords = np.array([
            (i, j, k) for k in (np.arange(self.mesh_size[2]+1,
                                          dtype='float64')*self.block_size[2])
            for j in (np.arange(self.mesh_size[1] + 1,
                                dtype='float64')*self.block_size[1])
            for i in (np.arange(self.mesh_size[0] + 1,
                                dtype='float64')*self.block_size[0])
        ], dtype='float64')
        return self.mb.create_vertices(coords.flatten())

    def _create_hexa(self, i, j, k,  verts, mesh):
        # TODO: - Should go on Common
        #       - Refactor this (????????)
                # (i, j, k)
        hexa = [verts[i + (j * (mesh[0] + 1)) +
                      (k * ((mesh[0] + 1) * (mesh[1] + 1)))],
                # (i+1, j, k)
                verts[(i + 1) + (j * (mesh[0] + 1)) +
                      (k * ((mesh[0] + 1) * (mesh[1] + 1)))],
                # (i+1, j+1, k)
                verts[(i + 1) + (j + 1) * (mesh[0]) +
                      (j + 1) + (k * ((mesh[0] + 1)*(mesh[1] + 1)))],
                # (i, j+1, k)
                verts[i + (j + 1) * (mesh[0]) + (j + 1) +
                      (k * ((mesh[0] + 1) * (mesh[1] + 1)))],
                # (i, j, k+1)
                verts[i + (j * (mesh[0] + 1)) +
                      ((k + 1) * ((mesh[0] + 1) * (mesh[1] + 1)))],
                # (i+1, j, k+1)
                verts[(i + 1) + (j * (mesh[0] + 1)) +
                      ((k + 1) * ((mesh[0] + 1) * (mesh[1] + 1)))],
                # (i+1, j+1, k+1)
                verts[(i + 1) + (j + 1) * (mesh[0]) +
                      (j + 1) + ((k + 1) * ((mesh[0] + 1) * (mesh[1] + 1)))],
                # (i, j+1, k+1)
                verts[i + (j + 1) * (mesh[0]) +
                      (j + 1) + ((k + 1) * ((mesh[0] + 1) * (mesh[1] + 1)))]]

        return hexa

    def create_fine_blocks_and_primal(self):
        # TODO: - Should go on Common
        fine_vertices = self.create_fine_vertices()
        cur_id = 0
        # Create fine grid
        for k, idz in zip(xrange(self.mesh_size[2]),
                          self.primal_ids[2]):
            # Flake8 bug
            print "{0} / {1}".format(k + 1, self.mesh_size[2])
            for j, idy in zip(xrange(self.mesh_size[1]),
                              self.primal_ids[1]):
                for i, idx in zip(xrange(self.mesh_size[0]),
                                  self.primal_ids[0]):

                    hexa = self._create_hexa(i, j, k,
                                             fine_vertices,
                                             self.mesh_size)
                    el = self.mb.create_element(types.MBHEX, hexa)

                    self.mb.tag_set_data(self.gid_tag, el, cur_id)
                    cur_id += 1
                    # Fine Global ID
                    self.mb.tag_set_data(self.gid_tag, el, cur_id)
                    # Fine Porosity
                    self.mb.tag_set_data(self.phi_tag, el, self.phi_values[
                        cur_id])
                    # Fine Permeability tensor
                    self.mb.tag_set_data(self.perm_tag, el, [
                        self.perm_values[cur_id], 0, 0,
                        0, self.perm_values[cur_id + self.mesh_size[0] *
                                            self.mesh_size[1] *
                                            self.mesh_size[2]], 0,
                        0, 0, self.perm_values[cur_id + 2*self.mesh_size[0] *
                                               self.mesh_size[1] *
                                               self.mesh_size[2]]])

                    self.elems.append(el)

                    # Create primal coarse grid
                    try:
                        primal = self.primals[(idx, idy, idz)]
                        self.mb.add_entities(primal, [el])
                        self.mb.tag_set_data(
                            self.fine_to_primal_tag, el, primal)
                    except KeyError:
                        primal = self.mb.create_meshset()
                        self.primals[(idx, idy, idz)] = primal
                        self.mb.add_entities(primal, [el])
                        self.mb.tag_set_data(
                            self.fine_to_primal_tag, el, primal)

        primal_id = 0
        for primal in self.primals.values():
            self.mb.tag_set_data(self.primal_id_tag, primal, primal_id)
            primal_id += 1

    def store_primal_adj(self):
        # TODO: - Should go on Common

        min_coarse_ids = np.array([0, 0, 0])
        max_coarse_ids = np.array([max(self.primal_ids[0]),
                                   max(self.primal_ids[1]),
                                   max(self.primal_ids[2])])

        for primal_id, primal in self.primals.iteritems():
            adj = self.mb.create_meshset()
            adj_ids = []

            for i in np.arange(-1, 2):
                for j in np.arange(-1, 2):
                    for k in np.arange(-1, 2):
                        coord_inc = np.array([i, j, k])
                        adj_id = primal_id + coord_inc
                        if any(adj_id != primal_id) and \
                           (sum(coord_inc == [0, 0, 0]) == 2) and \
                           all(adj_id >= min_coarse_ids) and \
                           all(adj_id <= max_coarse_ids):

                            self.mb.add_entities(
                                adj, [self.primals[tuple(adj_id)]])
                            adj_ids.append(tuple(adj_id))

            self.mb.tag_set_data(self.primal_adj_tag, primal, adj)

            self.primal_adj[primal_id] = adj_ids

    def _get_block_by_ijk(self, i, j, k):
        # TODO: - Should go on Common
        #       - Should reformulate to get self.mesh_size instead of input

        """
        Track down the block from its (i,j,k) position.
        """
        block = (k)*self.mesh_size[0]*self.mesh_size[1]+(
            (i)+(j)*self.mesh_size[0])
        return block

    def _get_elem_by_ijk(self, ijk):
        # TODO Should go on Common

        block_id = self._get_block_by_ijk(
            ijk[0], ijk[1], ijk[2])
        elem = self.elems[block_id]
        return elem

    def read_phi(self):
        # TODO: - Should go on Common
        #       - This should go on .cfg

        phi_values = []
        with open('spe_phi.dat') as phi:
            for line in phi:
                phi_values.extend(line.rstrip().split('        	'))
        self.phi_values = [float(val) for val in phi_values]

    def read_perm(self):
        # TODO: - Should go on Common
        #       - This should go on .cfg

        perm_values = []
        # i = 0
        with open('spe_perm.dat') as perm:
            for line in perm:
                line_list = line.rstrip().split('        	')
                if len(line_list) > 1:
                    perm_values.extend(line_list)
        self.perm_values = [float(val) for val in perm_values]

    def upscale_phi(self):
        for _, primal in self.primals.iteritems():
            # Calculate mean phi on primal
            fine_elems_in_primal = self.mb.get_entities_by_type(
                primal, types.MBHEX)
            fine_elems_phi_values = self.mb.tag_get_data(self.phi_tag,
                                                         fine_elems_in_primal)
            primal_mean_phi = fine_elems_phi_values.mean()
            # Store mean phi on the primal meshset and internal elements
            self.mb.tag_set_data(self.primal_phi_tag, primal, primal_mean_phi)

    def upscale_perm_mean(self, average_method):
        self.average_method = average_method
        basis = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        perm = []
        for primal_id, primal in self.primals.iteritems():

            fine_elems_in_primal = self.mb.get_entities_by_type(
                primal, types.MBHEX)
            fine_perm_values = self.mb.tag_get_data(self.perm_tag,
                                                    fine_elems_in_primal)
            primal_perm = [tensor.reshape(3, 3) for tensor in fine_perm_values]
            for dim in range(0, 3):
                perm = [(np.dot(np.dot(tensor, basis[dim]), basis[dim]))
                        for tensor in primal_perm]
                if average_method == 'Arithmetic':
                    primal_perm[dim] = np.mean(perm[dim])
                elif average_method == 'Geometric':
                    primal_perm[dim] = np.prod(np.asarray(
                        perm[dim]))**(1/len(np.asarray(perm[dim])))
                elif average_method == 'Harmonic':
                    primal_perm[dim] = len(np.asarray(
                        perm[dim]))/sum(1/np.asarray(perm[dim]))
                else:
                    print "Choose either Arithmetic, Geometric or Harmonic."
                    exit()

            self.mb.tag_set_data(self.primal_perm_tag, primal,
                                 [primal_perm[0], 0, 0,
                                  0, primal_perm[1], 0,
                                  0, 0, primal_perm[2]])

    def _primal_centroid(self, setid):
        coarse_sums = np.array(
            [[0, 0, 0],
             [0, 0, 1],
             [0, 1, 0],
             [0, 1, 1],
             [1, 0, 0],
             [1, 0, 1],
             [1, 1, 0],
             [1, 1, 1]]
        )
        primal_centroid = (
            (np.asarray(setid) + coarse_sums[0]) *
            np.array([self.coarse_ratio[0],
                      self.coarse_ratio[1],
                      self.coarse_ratio[2]]) +
            (np.asarray(setid) + coarse_sums[1]) *
            np.array([self.coarse_ratio[0],
                      self.coarse_ratio[1],
                      self.coarse_ratio[2]]) +
            (np.asarray(setid) + coarse_sums[2]) *
            np.array([self.coarse_ratio[0],
                      self.coarse_ratio[1],
                      self.coarse_ratio[2]]) +
            (np.asarray(setid) + coarse_sums[3]) *
            np.array([self.coarse_ratio[0],
                      self.coarse_ratio[1],
                      self.coarse_ratio[2]]) +
            (np.asarray(setid) + coarse_sums[4]) *
            np.array([self.coarse_ratio[0],
                      self.coarse_ratio[1],
                      self.coarse_ratio[2]]) +
            (np.asarray(setid) + coarse_sums[5]) *
            np.array([self.coarse_ratio[0],
                      self.coarse_ratio[1],
                      self.coarse_ratio[2]]) +
            (np.asarray(setid) + coarse_sums[6]) *
            np.array([self.coarse_ratio[0],
                      self.coarse_ratio[1],
                      self.coarse_ratio[2]]) +
            (np.asarray(setid) + coarse_sums[7]) *
            np.array([self.coarse_ratio[0],
                      self.coarse_ratio[1],
                      self.coarse_ratio[2]]))

        primal_centroid = primal_centroid // 8
        return primal_centroid

    def set_local_problem(self):  # Other parameters might go in as an input
        # create specific tags for setting local problems
        for primal_id, primal in self.primals.iteritems():
            boundary_val = 1.0
            for k in xrange(self.mesh_size[2]):
                for j in xrange(self.mesh_size[1]):
                    for i in xrange(self._coarse_dims()[0]):
                        self.mb.tag_set_data(self.boundary_tag,
                                             self._get_elem_by_ijk(
                                                 (i * self.coarse_ratio[0],
                                                  j, k)), boundary_val)
                        self.mb.tag_set_data(self.boundary_tag,
                                             self._get_elem_by_ijk(
                                              ((i + 1) * self.coarse_ratio[0]
                                               - 1, j, k)), boundary_val)
            boundary_val += 1.0

    def upscale_perm_flow_based(self):
        # TODO: - matrix assembly;
        #       - boundaries for experiments in each directions
        #       - k_eff = q/(-grad(p).A)
        pass

    def coarse_grid(self):
        # We should include a swithc for either printing coarse grid or fine
        # grid here that is fedy by the .cfg file.

        # if self.grid == 'coarse'
        """
        This will not delete primal grid information prevously calculated,
        since it is only looking for elements within the root_set that are
        MBHEX, whilst all props from primal grid are stored as meshsets
        """
        fine_grid = self.mb.get_entities_by_type(self.root_set, types.MBHEX)
        self.mb.delete_entities(fine_grid)
        coarse_vertices = self.create_coarse_vertices()
        coarse_dims = self._coarse_dims()
        cur_id = 0
        for k in xrange(coarse_dims[2]):
            print "{0} / {1}".format(k + 1, coarse_dims[2])
            for j in xrange(coarse_dims[1]):
                for i in xrange(coarse_dims[0]):

                    hexa = self._create_hexa(i, j, k,
                                             coarse_vertices,
                                             coarse_dims)
                    el = self.mb.create_element(types.MBHEX, hexa)

        # Assign coarse scale properties previously calculated
                    self.mb.tag_set_data(
                        self.coarse_gid_tag, el, cur_id)
                    self.mb.tag_set_data(self.primal_phi_tag, el,
                                         self.mb.tag_get_data(
                                             self.primal_phi_tag,
                                             self.primals[(i, j, k)]))
                    self.mb.tag_set_data(self.primal_perm_tag, el,
                                         self.mb.tag_get_data(
                                             self.primal_perm_tag,
                                             self.primals[(i, j, k)]))
                    if i == 0:
                        self.mb.tag_set_data(self.boundary_tag, el, 1.0)
                    if i == coarse_dims[0] - 1:
                        self.mb.tag_set_data(self.boundary_tag, el, 1.0)
                    self.coarse_elems.append(el)
                    cur_id += 1
    # def export(self, outfile):
    #     self.mb.write_file(outfile)
