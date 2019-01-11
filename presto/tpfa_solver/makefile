# Include the MOAB configuration information so that
# all required flags and libs are populated correctly
include makefile.config

default: all

# ALLEXAMPLES = tpfa_partitioning
ALLEXAMPLES = parallel_tpfa_solver
# ALLEXAMPLES = parallel_tpfa_solver_refactor

all: $(ALLEXAMPLES)

# tpfa_partitioning: tpfa_partitioning.o
parallel_tpfa_solver: parallel_tpfa_solver.o
# parallel_tpfa_solver_refactor: parallel_tpfa_solver_refactor.o
	@echo "[CXXLD]  $@"
	${VERBOSE}$(MOAB_CXX) -o $@ $< $(MOAB_LIBS_LINK) -std=c++11 -I/usr/include -I/usr/include -DMYAPP_EPETRA -L/usr/lib  -lpytrilinos -lamesos -laztecoo -ltrilinosss -ltriutils -lepetra -lteuchoskokkoscomm -lteuchoskokkoscompat -lteuchosremainder -lteuchosnumerics -lteuchoscomm -lteuchosparameterlist -lteuchoscore -lteuchoskokkoscomm -lteuchoskokkoscompat -lteuchosremainder -lteuchosnumerics -lteuchoscomm -lteuchosparameterlist -lteuchoscore -lkokkoscore -lkokkoscore /usr/lib/x86_64-linux-gnu/libdl.so /usr/lib/liblapack.so /usr/lib/libblas.so /usr/lib/x86_64-linux-gnu/libpthread.so

tpfa_partitioning: tpfa_partitioning.o
	@echo "[CXXLD]  $@"
	${VERBOSE}$(MOAB_CXX) -o $@ $< $(MOAB_LIBS_LINK) -std=c++11 -I/usr/include -I/usr/include -DMYAPP_EPETRA -L/usr/lib  -lpytrilinos -lamesos -laztecoo -ltrilinosss -ltriutils -lepetra -lteuchoskokkoscomm -lteuchoskokkoscompat -lteuchosremainder -lteuchosnumerics -lteuchoscomm -lteuchosparameterlist -lteuchoscore -lteuchoskokkoscomm -lteuchoskokkoscompat -lteuchosremainder -lteuchosnumerics -lteuchoscomm -lteuchosparameterlist -lteuchoscore -lkokkoscore -lkokkoscore /usr/lib/x86_64-linux-gnu/libdl.so /usr/lib/liblapack.so /usr/lib/libblas.so /usr/lib/x86_64-linux-gnu/libpthread.so

run: all $(addprefix run-,$(ALLEXAMPLES))

clean: clobber
	rm -rf ${ALLEXAMPLES}

# -lepetraext
