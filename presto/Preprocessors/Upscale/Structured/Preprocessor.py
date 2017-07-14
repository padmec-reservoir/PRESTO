import time
from StructuredUpscalingMethods import StructuredUpscalingMethods


class Preprocessor(object):

    def __init__(self, configs):
        self.configs = configs

        self.structured_configs = self.configs['StructuredUPS']

        self.values = self.structured_configs['coarse-ratio']
        self.coarse_ratio = [int(v) for v in self.values]

        self.values = self.structured_configs['mesh-size']
        self.mesh_size = [int(v) for v in self.values]

        self.values = self.structured_configs['block-size']
        self.block_size = [int(v) for v in self.values]

        self.method = self.structured_configs['method']
        if self.method == "Average":
            self.average = self.structured_configs['average']
        elif self.method == "Flow-based":
            self.average = self.structured_configs['average']
        else:
            print "Choose either Flow-based or Average."
            exit()

    def run(self, moab):

        self.SUM = StructuredUpscalingMethods(
            self.coarse_ratio, self.mesh_size, self.block_size, moab)
        self.SUM.calculate_primal_ids()
        self.SUM.create_tags()

        print "Creating fine vertices..."
        t0 = time.time()
        self.SUM.create_fine_vertices()
        print "took {0}".format(time.time() - t0), "seconds..."

        print "Reading porosity map..."
        t0 = time.time()
        self.SUM.read_phi()
        print "took {0}".format(time.time() - t0), "seconds..."

        print "Reading permeability map..."
        t0 = time.time()
        self.SUM.read_perm()
        print "took {0}".format(time.time() - t0), "seconds..."

        print "Creating fine blocks and associating to primal..."
        t0 = time.time()
        self.SUM.create_fine_blocks_and_primal()
        print "took {0}".format(time.time()-t0), "seconds..."

        print "Upscaling the porosity..."
        t0 = time.time()
        self.SUM.upscale_phi()
        print "took {0}".format(time.time()-t0), "seconds..."

        print "{0}".format(self.method), "upscaling for the permeability"

        if self.method == "Average":
            print "{0}".format(self.average), "mean..."
            t0 = time.time()
            self.SUM.upscale_perm_mean('Arithmetic')
            print "took {0}".format(time.time()-t0), "seconds..."

        if self.method == "Flow-based":
            print "Setting local boundaries..."
            t0 = time.time()
            self.SUM.set_local_problem()
            print "took {0}".format(time.time()-t0), "seconds..."

        print "Generating coarse scale grid..."
        t0 = time.time()
        self.SUM.set_local_problem()
        # self.SUM.coarse_grid()
        print "took {0}".format(time.time()-t0), "seconds..."
