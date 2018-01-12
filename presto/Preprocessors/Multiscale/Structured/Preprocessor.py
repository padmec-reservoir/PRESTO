import time

from .StructuredMultiscaleMesh import StructuredMultiscaleMesh


class Preprocessor(object):
    """
    Creates a 3-D structured grid and aggregates elements in primal and dual
    coarse entities.
    """

    def __init__(self, configs):
        self.configs = configs

        self.structured_configs = self.configs['StructuredMS']
        self.coarse_ratio = self.structured_configs['coarse-ratio']
        self.mesh_size = self.structured_configs['mesh-size']
        self.block_size = self.structured_configs['block-size']

        self.smm = StructuredMultiscaleMesh(
            self.coarse_ratio, self.mesh_size, self.block_size)

    def run(self, moab):
        self.smm.set_moab(moab)

        self.smm.calculate_primal_ids()
        self.smm.create_tags()

        print("Creating fine vertices...")
        t0 = time.time()
        self.smm.create_fine_vertices()
        print("took {0}\n".format(time.time()-t0))

        print("Creating fine blocks and primal...")
        t0 = time.time()
        self.smm.create_fine_blocks_and_primal()
        print("took {0}\n".format(time.time()-t0))

        print("Generating dual...")
        t0 = time.time()
        self.smm.generate_dual()
        self.smm.store_primal_adj()
        print("took {0}\n".format(time.time()-t0))

    @property
    def structured_configs(self):
        return self._structured_configs

    @structured_configs.setter
    def structured_configs(self, configs):
        if not configs:
            raise ValueError("Must have a [StructuredMS] section "
                             "in the config file.")

        self._structured_configs = configs

    @property
    def coarse_ratio(self):
        return self._coarse_ratio

    @coarse_ratio.setter
    def coarse_ratio(self, values):
        if not values:
            raise ValueError("Must have a coarse-ratio option "
                             "under the [StructuredMS] section in the config "
                             "file.")

        self._coarse_ratio = [int(v) for v in values]

    @property
    def mesh_size(self):
        return self._mesh_size

    @mesh_size.setter
    def mesh_size(self, values):
        if not values:
            raise ValueError("Must have a mesh-size option "
                             "under the [StructuredMS] section in the config "
                             "file.")

        self._mesh_size = [int(v) for v in values]

    @property
    def block_size(self):
        return self._block_size

    @block_size.setter
    def block_size(self, values):
        if not values:
            raise ValueError("Must have a block-size option "
                             "under the [StructuredMS] section in the config "
                             "file.")

        self._block_size = [int(v) for v in values]
