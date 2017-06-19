
<p align="center">
  <img src="https://cdn.rawgit.com/padmec-reservoir/PRESTO/master/PRESTO.png" width="700px"/>
</p>

# PRESTO
**PRESTO**: The Python REservoir Simulation Toolbox for flow simulation and diagnostics in porous media. PRESTO Currently deals with single phase fluid flow in highly heterogeneous and anisitropic porous media in two or three dimensions. So far, PRESTO's main features are:
* Classic Multiscale simulation;
* Averaging Upscaling.

It is built on top of Python, and uses the [PyMoab](https://bitbucket.org/fathomteam/moab/overview) and [PyTrilinos](https://github.com/trilinos/Trilinos) libraries to handle the internal mesh data structure, and matrix solving, respectively.

Currently, PRESTO only runs on Python 2.7, since PyTrilinos only supports this version. Also, parallelism through MPI4Py is stale for now, since the PyMoab doesn't yet support it.

# Dependencies
* [PyMoab](https://bitbucket.org/fathomteam/moab/overview)
* [PyTrilinos](https://github.com/trilinos/Trilinos)

# Documentation
Coming soon.








