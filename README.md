
<p align="center">
  <img src="https://cdn.rawgit.com/padmec-reservoir/PRESTO/master/PRESTO.png" width="700px"/>
</p>

---

# Description
**PRESTO**: The Python REservoir Simulation Toolbox for flow simulation and diagnostics in porous media. PRESTO Currently deals with single phase fluid flow in highly heterogeneous and anisitropic porous media in two or three dimensions. So far, PRESTO's main features are:
* Preprocessor for threedimensional Classic Multiscale simulation;
* Some techniques for upscaling reservoir properties;
* Fine grid and Coarse grid generation for structured TPFA simulations.

It is built on top of Python, and uses the [ELLIPTIc](https://github.com/ricardolira/ELLIPTIc) library to handle the internal mesh data structure, and matrix solving.

Currently, PRESTO only runs on Python 2.7, since PyTrilinos only supports this version. Also, parallelism through MPI4Py is stale for now, since the PyMoab doesn't yet support it.

# Dependencies
* [ELLIPTIc](https://github.com/padmec-reservoir/ELLIPTIc)


# Documentation
Coming soon.








