import time
from StructuredUpscalingMethods import StructuredUpscalingMethods


class Preprocessor(object):

    def __init__(self, configs):
        self.configs = configs

        self.structured_general = self.configs['General']

        self.output_file = self.structured_general['output-file']
        self.fine_grid_construct = self.structured_general['fine-grid']

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
            if self.average not in ('Arithmetic', 'Geometric', 'Harmonic'):
                print "Choose either Arithmetic, Geometric or Harmonic."
                exit()
        elif self.method != 'Flow-based':
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

        print "Associating fine volumes to primal coarse grid..."
        t0 = time.time()
        self.SUM.create_fine_blocks_and_primal()
        print "took {0}".format(time.time()-t0), "seconds..."

        if self.fine_grid_construct == 'fine_grid':
            print "exporting fine scale mesh"
            t0 = time.time()
            self.SUM.export(self.output_file)
            print "took {0}".format(time.time()-t0), "seconds..."
            exit()

        print "Upscaling the porosity..."
        t0 = time.time()
        self.SUM.upscale_phi()
        print "took {0}".format(time.time()-t0), "seconds..."

        print "{0}".format(self.method), "upscaling for the permeability"

        if self.method == "Average":
            print "{0}".format(self.average), "mean..."
            t0 = time.time()
            self.SUM.upscale_perm_mean(self.average)
            print "took {0}".format(time.time()-t0), "seconds..."

        if self.method == "Flow-based":
            print "Setting local boundaries..."
            t0 = time.time()
            self.SUM.set_local_problem()
            print "took {0}".format(time.time()-t0), "seconds..."

        print "Generating coarse scale grid..."
        t0 = time.time()
        # self.SUM.coarse_grid()
        self.SUM.upscale_perm_flow_based()
        print "took {0}".format(time.time()-t0), "seconds..."

        print "Exporting..."
        t0 = time.time()
        self.SUM.export(self.output_file)
        print "took {0}\n".format(time.time()-t0)
