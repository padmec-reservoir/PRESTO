#!/bin/sh
echo "#### CREATING MESH ####"
python simple_mesh_generator.py 60 20 220 10 85 2
echo "#### DONE ####"

echo "#### PARTITIONING MESH ####"
make tpfa_partitioning && ./tpfa_partitioning
echo "#### DONE ####"

echo "#### RUNNING TPFA ####"
make parallel_tpfa_solver && mpiexec -np 4 ./parallel_tpfa_solver
echo "#### DONE ####"

