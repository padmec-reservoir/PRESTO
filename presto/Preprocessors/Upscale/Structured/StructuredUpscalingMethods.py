import numpy as np
from pymoab import core
from pymoab import types
from pymoab import topo_util


class StructuredUpscalingMethods:
    """Defines a structured upscaling mesh representation

    Parameters
    ----------
    coarse_ratio: List or array of integers
        List or array containing three values indicating the coarsening ratio
        of the mesh in x, y and z.
        mesh_size: List or array of integers
            List or array containing three values indicating the mesh size
            (number of fine elements) of the mesh in x, y and z.
        block_size List o array of floats
            List or array containing three values indicating the constant
            increments of vertex coordinates in x, y and z.
        """
    def __init__(self, coarse_ratio, mesh_size, block_size, moab):
        # TODO Should go on Common

        self.coarse_ratio = coarse_ratio
        self.mesh_size = mesh_size
        self.block_size = block_size
        self.block_size_coarse = []

        self.verts = None  # Array containing MOAB vertex entities
        self.elems = []  # List containing MOAB volume entities

        self.coarse_verts = None  # Array containing MOAB vertex entities for
        # the coarse mesh
        self.coarse_elems = []  # List containig MOAB volume entities for the
        # coarse mesh

        self.primals = {}  # Mapping from tuples (idx, dy, idz) to Coarse
        # volumes
        self.primal_ids = []

        self.primals_adj = []

        self.perm = []

        # MOAB boilerplate
        self.mb = moab
        self.root_set = self.mb.get_root_set()
        self.mesh_topo_util = topo_util.MeshTopoUtil(self.mb)

    def _block_size_coarse(self, mesh_size, block_size, coarse_ratio):
        # Should go on Common

        total_size = np.asarray(self.mesh_size, dtype='int32') * np.asarray(
            self.block_size, dtype='float64')
        mesh_size_coarse = np.asarray(
            self.mesh_size, dtype='int32') // np.asarray(
                self.coarse_ratio, dtype='int32')

        for dim in range(0, 3):
            self.block_size_coarse.append([self.coarse_ratio[dim]*np.asarray(
                self.block_size[dim], dtype='float64') * self.mesh_size[dim]
                for self.mesh_size[dim] in np.arange(mesh_size_coarse[dim],
                                                     dtype='int32')])
            self.block_size_coarse[dim].append(total_size[dim])
        return self.block_size_coarse

    def _coarse_dims(self, mesh_size, coarse_ratio):
        # TODO Should go on Common

        mesh_size_coarse = np.asarray(
            self.mesh_size, dtype='int32') // np.asarray(
                self.coarse_ratio, dtype='int32')
        return mesh_size_coarse

    def coarse_coords():
        # TODO Should go on Common

        coarse_coords = np.array([
            (i, j, k) for k in (np.array(block_size_z_coarse, dtype='float64'))
            for j in (np.array(block_size_y_coarse, dtype='float64'))
            for i in (np.array(block_size_x_coarse, dtype='float64'))
        ])

    def calculate_primal_ids(self):
        # TODO Should go on Common

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
                    dim] // self.coarse_ratio[dim] * self.coarse_ratio[dim]]
                                        + new_primal[dim])



    def create_fine_vertices(self):
        # TODO Should go on Common

        max_mesh_size = max(
            self.mesh_size[2]*self.block_size[2],
            self.mesh_size[1]*self.block_size[1],
            self.mesh_size[0]*self.block_size[0])

        coords = np.array([(i, j, k)
                           for k in (
                               np.arange(
                                   self.mesh_size[2]+1, dtype='float64') *
                               self.block_size[2]/max_mesh_size)
                           for j in (
                               np.arange(
                                   self.mesh_size[1]+1, dtype='float64') *
                               self.block_size[1]/max_mesh_size)
                           for i in (
                               np.arange(
                                   self.mesh_size[0]+1, dtype='float64') *
                               self.block_size[0]/max_mesh_size)
                           ], dtype='float64')
        self.verts = self.mb.create_vertices(coords.flatten())

    def create_tags(self):
        # TODO Should go on Common (?)

        self.gid_tag = self.mb.tag_get_handle(
            "GLOBAL_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_DENSE, True)

        self.primal_id_tag = self.mb.tag_get_handle(
            "PRIMAL_ID", 1, types.MB_TYPE_INTEGER, types.MB_TAG_SPARSE, True)

        self.phi_tag = self.mb.tag_get_handle(
            "PHI", 1, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

        self.primal_phi_tag = self.mb.tag_get_handle(
            "PRIMAL_PHI", 1, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

        self.perm_tag = self.mb.tag_get_handle(
            "PERM", 9, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

        self.primal_perm_tag = self.mb.tag_get_handle(
            "PRIMAL_PERM", 9, types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE, True)

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

    def _create_hexa(self, i, j, k):
        # TODO Should go on Common
        # TODO: Refactor this (????????)
                # (i, j, k)
        hexa = [self.verts[(i)+(j*(self.mesh_size[0]+1))+(k*((
            self.mesh_size[0]+1)*(self.mesh_size[1]+1)))],
                # (i+1, j, k)
                self.verts[(i+1)+(j*(self.mesh_size[0]+1))+(k*((
                    self.mesh_size[0]+1)*(self.mesh_size[1]+1)))],
                # (i+1, j+1, k)
                self.verts[(i+1)+(j+1)*(self.mesh_size[0])+(j+1)+(k*((
                    self.mesh_size[0]+1)*(self.mesh_size[1]+1)))],
                # (i, j+1, k)
                self.verts[(i)+(j+1)*(self.mesh_size[0])+(j+1)+(k*((
                    self.mesh_size[0]+1)*(self.mesh_size[1]+1)))],
                # (i, j, k+1)
                self.verts[(i)+(j*(self.mesh_size[0]+1))+((k+1)*((
                    self.mesh_size[0]+1)*(self.mesh_size[1]+1)))],
                # (i+1, j, k+1)
                self.verts[(i+1)+(j*(self.mesh_size[0]+1))+((k+1)*((
                    self.mesh_size[0]+1)*(self.mesh_size[1]+1)))],
                # (i+1, j+1, k+1)
                self.verts[(i+1)+(j+1)*(self.mesh_size[0])+(j+1)+((k+1)*((
                    self.mesh_size[0]+1)*(self.mesh_size[1]+1)))],
                # (i, j+1, k+1)
                self.verts[(i)+(j+1)*(self.mesh_size[0])+(j+1)+((k+1)*((
                    self.mesh_size[0]+1)*(self.mesh_size[1]+1)))]]

        return hexa

    def create_fine_blocks_and_primal(self):
        # TODO Should go on Common

        cur_id = 0
        # Create fine grid
        for k, idz in zip(xrange(self.mesh_size[2]),
                          self.primal_ids[2]):

            print "{0} / {1}".format(k, self.mesh_size[2])

            for j, idy in zip(xrange(self.mesh_size[1]),
                              self.primal_ids[1]):

                for i, idx in zip(xrange(self.mesh_size[0]),
                                  self.primal_ids[0]):

                    hexa = self._create_hexa(i, j, k)
                    el = self.mb.create_element(types.MBHEX, hexa)

                    self.mb.tag_set_data(self.gid_tag, el, cur_id)
                    cur_id += 1
                    self.mb.tag_set_data(self.gid_tag, el, cur_id)
                    self.mb.tag_set_data(self.phi_tag, el, self.phi_values[
                        cur_id])
                    self.mb.tag_set_data(self.perm_tag, el, [
                        self.perm_values[cur_id], 0, 0,
                        0, self.perm_values[
                            cur_id+self.mesh_size[0] * self.mesh_size[1] *
                            self.mesh_size[2]], 0, 0, 0, self.perm_values[
                                cur_id+2*self.mesh_size[0] *
                                self.mesh_size[1] * self.mesh_size[2]]])

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
        # TODO Should go on Common

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

    def _get_block_by_ijk(self, i, j, k, n_i, n_j):
        # TODO Should go on Common

        """
        Track down the block from its (i,j,k) position.
        """
        block = (k)*n_i*n_j+((i)+(j)*n_i)
        return block

    def _get_elem_by_ijk(self, ijk):
        # TODO Should go on Common

        block_id = self._get_block_by_ijk(
            ijk[0], ijk[1], ijk[2], self.mesh_size[0], self.mesh_size[1])
        elem = self.elems[block_id]
        return elem

    def read_phi(self):
        # TODO Should go on Common

        phi_values = []
        with open('spe_phi.dat') as phi:
            for line in phi:
                phi_values.extend(line.rstrip().split('        	'))
        self.phi_values = [float(val) for val in phi_values]

    def read_perm(self):
        # TODO Should go on Common

        perm_values = []
        # i = 0
        with open('spe_perm.dat') as perm:
            for line in perm:
                line_list = line.rstrip().split('        	')
                if len(line_list) > 1:
                    perm_values.extend(line_list)
        self.perm_values = [float(val) for val in perm_values]

    def upscale_phi(self):
        for primal_id, primal in self.primals.iteritems():
            # Calculate mean phi on primal
            fine_elems_in_primal = self.mb.get_entities_by_type(
                primal, types.MBHEX)
            fine_elems_phi_values = self.mb.tag_get_data(self.phi_tag,
                                                         fine_elems_in_primal)
            primal_mean_phi = fine_elems_phi_values.mean()
            # Store mean phi on the primal meshset and internal elements
            self.mb.tag_set_data(self.primal_phi_tag, primal, primal_mean_phi)
            self.mb.tag_set_data(self.primal_phi_tag, fine_elems_in_primal,
                                 np.repeat(primal_mean_phi, len(
                                     fine_elems_in_primal)))

    def upscale_perm_mean(self, average_method):
        self.average_method = average_method
        primal_perm = {}
        # perm = []
        basis = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

        for primal_id, primal in self.primals.iteritems():

            fine_elems_in_primal = self.mb.get_entities_by_type(
                primal, types.MBHEX)
            fine_perm_values = self.mb.tag_get_data(self.perm_tag,
                                                    fine_elems_in_primal)
            primal_perm.update({eid: tensor.reshape(3, 3) for eid, tensor in
                                zip(fine_elems_in_primal, fine_perm_values)
                                })
# fix this part to perform averaging in less code lines
            for dim in range(0, 3):
                self.perm.append([(np.dot(np.dot(primal_perm[elem_id],
                                                 basis[dim]), basis[dim]))
                                  for elem_id in fine_elems_in_primal])
                if average_method == 'Arithmetic':
                    primal_perm[dim] = np.mean(self.perm[dim])
                if average_method == 'Geometric':
                    primal_perm[dim] = np.prod(np.asarray(
                        self.perm[dim]))**(1/len(np.asarray(self.perm[dim])))
                if average_method == 'Harmonic':
                    primal_perm[dim] = len(np.asarray(
                        self.perm[dim]))/sum(1/np.asarray(self.perm[dim]))
            self.mb.tag_set_data(self.primal_perm_tag, primal,
                                 [primal_perm[0], 0, 0,
                                  0, primal_perm[1], 0,
                                  0, 0, primal_perm[2]])

    def coarse_grid(self):

        dim = self._coarse_dims(self.mesh_size, self.coarse_ratio)

        print self.mesh_size, self.block_size, self.coarse_ratio
        total_size_coarse = self._block_size_coarse(self.mesh_size,
                                                    self.block_size,
                                                    self.coarse_ratio)
        print total_size_coarse
        for k in xrange(dim[2]):

            print round(float(k+1)/float(dim[2]) * 100, 2), ' %'

            for j in xrange(dim[1]):

                for i in xrange(dim[0]):

                    hexa = self._create_hexa(i, j, k)
                    el = self.mb.create_element(types.MBHEX, hexa)

        # Assign coarse scale properties
"""
                    #Create TAG
                    self.mb.tag_set_data(self.coarse_gid_tag,
                                         el, cur_id)

                    self.mb.tag_set_data(self.primal_phi_tag, el,
                                         coarse_mean_phi[cur_id])

        #How to assure information correctly assignment???


                    self.mb.tag_set_data(self.primal_perm_tag, el,
                                         [coarse_perm_xx[cur_id], 0, 0,
                                                   0, coarse_perm_yy[cur_id], 0,
                                                   0, 0, coarse_perm_zz[cur_id]])
                    cur_id += 1
                    coarse_elems.append(el)

"""
    #    pass
