
COMMON_COMMAND=-arch=sm_50 -rdc=true -lcudadevrt -lcufft --use_fast_math -O4 -std=c++11
compiler=nvcc

all: SPONGE SPONGE_FEP SPONGE_TI
	
install: all
	
MD_OBJECTS = main.o common.o control.o MD_core/MD_core.o bond/bond.o bond/bond_soft.o angle/angle.o angle/Urey_Bradley_force.o dihedral/dihedral.o dihedral/improper_dihedral.o cmap/cmap.o nb14/nb14.o neighbor_list/neighbor_list.o Lennard_Jones_force/Lennard_Jones_force.o Lennard_Jones_force/LJ_soft_core.o PME_force/PME_force.o  thermostats/Langevin_MD.o thermostats/Middle_Langevin_MD.o thermostats/Andersen_thermostat.o thermostats/Berendsen_thermostat.o thermostats/nose_hoover_chain.o barostats/MC_barostat.o barostats/Berendsen_barostat.o barostats/andersen_barostat.o restrain/restrain.o constrain/constrain.o constrain/simple_constrain.o constrain/SETTLE.o constrain/SHAKE.o virtual_atoms/virtual_atoms.o crd_molecular_map/crd_molecular_map.o
include $(MD_OBJECTS:.o=.d)

SPONGE: $(MD_OBJECTS)
	$(compiler) -o $@  $^ $(COMMON_COMMAND)
		

FEP_OBJECTS = main_fep.o common.o control.o FEP_core/FEP_core.o bond/bond.o bond/bond_soft.o angle/angle.o dihedral/dihedral.o nb14/nb14.o neighbor_list/neighbor_list.o  Lennard_Jones_force/LJ_soft_core.o PME_force/PME_force.o
include $(FEP_OBJECTS:.o=.d)

SPONGE_FEP: $(FEP_OBJECTS)
	$(compiler) -o $@  $^ $(COMMON_COMMAND)

TI_OBJECTS = main_ti.o common.o control.o TI_core/TI_core.o bond/bond.o bond/bond_soft.o angle/angle.o dihedral/dihedral.o nb14/nb14.o neighbor_list/neighbor_list.o  Lennard_Jones_force/LJ_soft_core.o PME_force/PME_force.o
include $(TI_OBJECTS:.o=.d)

SPONGE_TI: $(TI_OBJECTS)
	$(compiler) -o $@  $^ $(COMMON_COMMAND)

%.d: %.cu
	$(info )
	$(info analyzing dependency of $<)
	set -e; rm -f $@; \
	$(compiler) -MM $(CPPFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

%.o: %.cu
	$(compiler) -o $@ -c $< $(COMMON_COMMAND)

SUBDIRS=$(shell ls -l | grep ^d | awk '{print $$9}')
clean:
	rm -f *.d
	rm -f *.o
	rm -f $(foreach i, $(SUBDIRS), $(i)/*.o)
	rm -f $(foreach i, $(SUBDIRS), $(i)/*.d)
	rm -f $(foreach i, $(SUBDIRS), $(i)/*.d.*)
